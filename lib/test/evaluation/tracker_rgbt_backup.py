import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler


def trackerlist(
    name: str, parameter_name: str, dataset_name: str, run_ids=None, display_name: str = None, result_only=False, save_suffix=""
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
        parameter_name: str,
        dataset_name: str,
        run_id: int = None,
        display_name: str = None,
        result_only=False,
        tracker_params=None,
        debug=None,
        save_suffix="",
    ):
        # assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            if save_suffix != "":
                self.results_dir = "{}/{}/{}_{}".format(env.results_path, self.name, self.parameter_name, save_suffix)
            else:
                self.results_dir = "{}/{}/{}".format(env.results_path, self.name, self.parameter_name)
        else:
            if save_suffix != "":
                self.results_dir = "{}/{}/{}_{}_{}".format(env.results_path, self.name, self.parameter_name, save_suffix, self.run_id)
            else:
                self.results_dir = "{}/{}/{}_{}".format(env.results_path, self.name, self.parameter_name, self.run_id)
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
        # params = self.get_parameters()
        # params = self.params

        # debug_ = debug
        # if debug is None:
        #     debug_ = getattr(params, 'debug', 0)

        # params.debug = debug_

        # Get init information
        init_info = seq.init_info()
        # tracker = self.create_tracker(params)

        output = self._track_sequence(seq, init_info)
        return output

    def _track_sequence(self, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i
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

        # Initialize
        image_v = self._read_image(seq.frames[0][0])
        image_i = self._read_image(seq.frames[0][1])

        start_time = time.time()
        out = self.tracker.initialize([image_v, image_i], init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {"target_bbox": init_info.get("init_bbox")[0], "time": time.time() - start_time}
        if self.tracker.params.save_all_boxes:
            init_default["all_boxes"] = out["all_boxes"]
            init_default["all_scores"] = out["all_scores"]

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image_v = self._read_image(frame_path[0])
            image_i = self._read_image(frame_path[1])

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info["previous_output"] = prev_output

            out = self.tracker.track([image_v, image_i], info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {"time": time.time() - start_time})

            # just debug
            # print("喵喵喵")
            # bbox = out["target_bbox"]
            # print(bbox)
            # image_v_show = cv.cvtColor(image_v, cv.COLOR_BGR2RGB)
            # cv.rectangle(
            #     image_v_show,
            #     [bbox[0], bbox[1]],
            #     [bbox[0] + bbox[2], bbox[1] + bbox[3]],
            #     color=(0, 0, 255),
            #     thickness=1,
            # )
            # cv.imshow("test", image_v_show)
            # cv.waitKey(0)

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
        params = param_module.parameters(self.parameter_name, model, search_area_scale)
        if tracker_params is not None:
            for param_k, v in tracker_params.items():
                setattr(params, param_k, v)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")


class Video(Dataset):
    def __init__(self, frame_vi_list) -> None:
        super().__init__()
        self.frame_vi_list = frame_vi_list

    def __len__(self):
        return len(self.frame_vi_list)

    def read_image(self, image_file):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)

    def __getitem__(self, index):
        image_v = self.read_image(self.frame_vi_list[index][0])
        image_i = self.read_image(self.frame_vi_list[index][1])
        return (image_v, image_i)
