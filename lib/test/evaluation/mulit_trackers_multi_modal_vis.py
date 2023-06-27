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

from lib.test.tracker.mulit_tracker_vis import Multi_Trackers
from lib.test.evaluation.data import RGBT_Sequence
from lib.test.utils import TrackerParams
from typing import List


class Vis_Trackers:
    """
    同时以augmentation crop (类似训练)的方式推理多个trackers (不同模型)
    用于可视化搜索帧特征, 用于比较特征提取效果

    Wraps the tracker for evaluation and running purposes.
    args:
        name: List of name of tracking method.
        training_parameter_name: List of name of training parameter file.
        tracking_parameter_name: 保证所有tracker使用相同的跟踪超参数
        model_name: List of checkpoint
    """

    def __init__(
        self,
        name: List[str],
        training_parameter_name: List[str],
        model_name: List[str],
        dataset_name: str,
        tracker_params=None,  # 可能的奇怪的跟踪参数，暂时作为保留项，我也不知道有啥用。。。
        save_suffix="",  # 添加保存后缀?, 前缀是tracking_parameter_name, 后缀用于保存training阶段相关
        tracking_parameter_name: str = "tracking_vis",  # ./experiments/xxxx.yaml for tracking hyper parameters, should use same for different trackers
    ):
        # assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.training_parameter_name = training_parameter_name
        self.tracking_parameter_name = tracking_parameter_name
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.trackers = None
        env = env_settings()
        if save_suffix != "":
            self.results_dir = "{}/{}_{}".format(env.results_path, "search_vis", save_suffix)
        else:
            self.results_dir = "{}/{}".format(env.results_path, "search_vis")
        self.params: List[TrackerParams] = self.get_parameters(tracker_params)

    def __len__(self):
        return len(self.name)

    def create_tracker(self):
        self.trackers = Multi_Trackers(self.name, self.params, self.dataset_name)

    def run_sequence(self, seq):
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

    def _track_sequence(self, seq: RGBT_Sequence, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        output = []  # search_vis_image

        video_dataset = Video(seq.frames)
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
                self.trackers.initialize(image_vi, init_info)
            else:
                if seq.dataset == "VTUAV":
                    if idx % 10 == 0:
                        info = seq.get_bbox(idx // 10)
                    else:
                        continue
                else:
                    info = seq.get_bbox(idx)
                out = self.trackers.track(image_vi, info)
                output.append(out)
        return output

    def get_parameters(self, tracker_params=None):
        """Get parameters."""
        params_list = []
        for name_tmp, train_param_tmp, model_temp_tmp in zip(self.name, self.training_parameter_name, self.model_name):
            param_module = importlib.import_module("lib.test.parameter.{}".format(name_tmp))
            params = param_module.parameters(train_param_tmp, self.tracking_parameter_name, model_temp_tmp)
            if tracker_params is not None:
                for param_k, v in tracker_params.items():
                    setattr(params, param_k, v)
            params_list.append(params)
        return params_list


class Video(Dataset):
    def __init__(self, frame_vi_list) -> None:
        super().__init__()
        self.frame_vi_list = frame_vi_list

    def __len__(self):
        return len(self.frame_vi_list)

    def read_image(self, image_file):
        im = cv2.imread(image_file)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        image_v = self.read_image(self.frame_vi_list[index][0])
        image_i = self.read_image(self.frame_vi_list[index][1])
        return (image_v, image_i)


def collect_image_single(batch):
    return batch[0]
