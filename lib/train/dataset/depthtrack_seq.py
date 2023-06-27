import os
import os.path
import torch
import numpy as np
import pandas

from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import glob
from lib.train.dataset.depth_utils import get_rgbd_frame


class DepthTrack(BaseVideoDataset):
    """
    https://github.com/xiaozai/DeT/tree/main 里提到 toy07_indoor_320 这个视频 some gt missing
    """

    def __init__(self, root=None, split="train", image_loader=jpeg4py_loader):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        self.root = env_settings().depthtrack_dir if root is None else root
        super().__init__("DepthTrack", self.root, image_loader)

        # Keep a list of all classes
        self.sequence_list = sorted(glob.glob(os.path.join(self.root, split, "*", "*")))  # 它多套了一层视频名的文件夹

        self.sequence_imgpath_list = []
        for seq_path in self.sequence_list:
            img_v_list = sorted(glob.glob(os.path.join(seq_path, "color", "*")))
            img_i_list = sorted(glob.glob(os.path.join(seq_path, "depth", "*")))
            self.sequence_imgpath_list.append(list(zip(img_v_list, img_i_list)))  # 存放索引对应各自模态文件名(因为文件名不一定是按bbox索引对应的....)

        # print(self.sequence_imgpath_list)
        # print(len(self.sequence_imgpath_list))
        # exit()
        self.sequence_info_list = []
        for seq_path in self.sequence_list:
            info_dict = self._get_sequence_info_(seq_path)
            if "toy07_indoor_320" in seq_path:
                for k in info_dict:
                    info_dict[k] = info_dict[k][:1367]  # 神奇的gt missing错误，就不能把部分label删了吗
            self.sequence_info_list.append(info_dict)

    def _get_sequence_info_(self, seq_path):
        bbox = self._read_bb_anno_(seq_path)  # [v_tensor, i_tensor]
        valid = (bbox[:, 0, 2] > 0) & (bbox[:, 0, 3] > 0) & (bbox[:, 1, 2] > 0) & (bbox[:, 1, 3] > 0)
        visible = valid.clone().byte()
        return {"bbox": bbox, "valid": valid, "visible": visible}

    def get_name(self):
        return "DepthTrack"

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno_(self, seq_path):
        gt_v = pandas.read_csv(
            os.path.join(seq_path, "groundtruth.txt"), delimiter=",", header=None, dtype=np.float32, na_filter=True, low_memory=False
        ).values
        gt_v = np.nan_to_num(gt_v)
        gt_i = gt_v.copy()
        gt_vi = torch.cat([torch.tensor(gt_v).unsqueeze(1), torch.tensor(gt_i).unsqueeze(1)], dim=1)  # (N, 2, 4) xywh
        return gt_vi

    def get_sequence_info(self, seq_id):
        return self.sequence_info_list[seq_id]

    def _get_frame(self, seq_id, frame_id):
        img_v_path, img_i_path = self.sequence_imgpath_list[seq_id][frame_id]
        img_v, img_d = get_rgbd_frame(img_v_path, img_i_path, dtype="rgb3d", depth_clip=True)
        return [img_v, img_d]

    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = [self._get_frame(seq_id, f_id) for f_id in frame_ids]  # [N,2,(H,W,C)]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]  # [N, (2, 4)] or (N, 1)

        # object_meta = OrderedDict(
        #     {"object_class_name": None, "motion_class": None, "major_class": None, "root_class": None, "motion_adverb": None}
        # )
        object_meta = OrderedDict()
        return frame_list, anno_frames, object_meta
