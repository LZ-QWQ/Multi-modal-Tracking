import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from lib.train.dataset.depth_utils import get_rgbd_frame


def trackerlist(
    name: str,
    parameter_name: str,
    dataset_name: str,
    run_ids=None,
    display_name: str = None,
    result_only=False,
    save_suffix="",
):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [
        RGBT_Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only, save_suffix=save_suffix) for run_id in run_ids
    ]


class RGBT_Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(
        self,
        name: str,
        training_parameter_name: str,
        dataset_name: str,
        run_id: int = None,
        display_name: str = None,
        result_only=False,
        tracker_params=None,
        debug=None,
        save_suffix="",  # 添加保存后缀?, 前缀是tracking_parameter_name, 后缀用于保存training阶段相关
        tracking_parameter_name: str = "tracking",  # ./experiments/xxxx.yaml for tracking hyper parameters
    ):
        # assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.training_parameter_name = training_parameter_name
        self.tracking_parameter_name = tracking_parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            if save_suffix != "":
                self.results_dir = "{}/{}/{}_{}".format(env.results_path, self.name, self.tracking_parameter_name, save_suffix)
            else:
                self.results_dir = "{}/{}/{}".format(env.results_path, self.name, self.tracking_parameter_name)
        else:
            if save_suffix != "":
                self.results_dir = "{}/{}/{}_{}_{}".format(
                    env.results_path, self.name, self.tracking_parameter_name, save_suffix, self.run_id
                )
            else:
                self.results_dir = "{}/{}/{}_{}".format(env.results_path, self.name, self.tracking_parameter_name, self.run_id)
        if result_only:
            raise NotImplementedError()
            self.results_dir = "{}/{}".format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tracker", "%s.py" % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module("lib.test.tracker.{}".format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.params = self.get_parameters(tracker_params)

    def create_tracker(self, device_id, debug=None):  # , params
        debug_ = debug
        if debug is None:
            debug_ = getattr(self.params, "debug", 0)
        self.params.debug = debug_

        self.tracker = self.tracker_class(self.params, self.dataset_name)
        self.tracker.network.cuda(device_id)
        # return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        # Get init information
        init_info = seq.init_info()
        output = self._track_sequence(seq, init_info)
        return output

    def _track_sequence(self, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        output = {"target_bbox": [], "time": []}
        if self.tracker.params.save_all_boxes:
            output["all_boxes"] = []
            output["all_scores"] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        video_dataset = Video(seq.frames, seq.dataset in ["DepthTrack"])  # RGBD数据集要单独处理读取，（因为深度图稍微没做过预处理呢）
        video_loader = DataLoader(
            video_dataset,
            batch_size=1,
            shuffle=False,
            sampler=SequentialSampler(video_dataset),
            drop_last=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collect_image_single,
            prefetch_factor=2,
        )

        for idx, image_vi in enumerate(video_loader):
            if idx == 0:
                # Initialize
                start_time = time.time()
                out = self.tracker.initialize(image_vi, init_info)
                if out is None:
                    out = {}
                prev_output = OrderedDict(out)
                init_default = {"target_bbox": init_info.get("init_bbox")[0], "time": time.time() - start_time}
                if self.tracker.params.save_all_boxes:
                    init_default["all_boxes"] = out["all_boxes"]
                    init_default["all_scores"] = out["all_scores"]
                _store_outputs(out, init_default)

            else:
                start_time = time.time()

                info = seq.frame_info(idx)
                info["previous_output"] = prev_output

                out = self.tracker.track(image_vi, info)
                prev_output = OrderedDict(out)
                _store_outputs(out, {"time": time.time() - start_time})

        for key in ["target_bbox", "all_boxes", "all_scores"]:
            if key in output and len(output[key]) <= 1:
                output.pop(key)
        return output

    def get_parameters(self, tracker_params=None):
        """Get parameters."""

        param_module = importlib.import_module("lib.test.parameter.{}".format(self.name))
        search_area_scale = None
        if tracker_params is not None and "search_area_scale" in tracker_params:
            search_area_scale = tracker_params["search_area_scale"]
        model = ""
        if tracker_params is not None and "model" in tracker_params:
            model = tracker_params["model"]
        params = param_module.parameters(self.training_parameter_name, self.tracking_parameter_name, model, search_area_scale)
        if tracker_params is not None:
            for param_k, v in tracker_params.items():
                setattr(params, param_k, v)
        return params


class Video(Dataset):
    def __init__(self, frame_vi_list, isdepth=False) -> None:
        super().__init__()
        self.frame_vi_list = frame_vi_list
        self.isdepth = isdepth

    def __len__(self):
        return len(self.frame_vi_list)

    def read_image(self, image_file):
        im = cv2.imread(image_file)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        if self.isdepth: # 深度读取稍微不太一样
            image_v, image_i = get_rgbd_frame(self.frame_vi_list[index][0], self.frame_vi_list[index][1], dtype="rgb3d", depth_clip=True)
        else:
            image_v = self.read_image(self.frame_vi_list[index][0])
            image_i = self.read_image(self.frame_vi_list[index][1])
        return (image_v, image_i)


def collect_image_single(batch):
    return batch[0]
