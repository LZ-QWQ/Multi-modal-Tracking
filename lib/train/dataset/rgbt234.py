import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import glob


class RGBT234(BaseVideoDataset):
    """

    """

    def __init__(self, root=None, image_loader=jpeg4py_loader):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        self.root = env_settings().rgbt234_dir if root is None else root
        super().__init__("RGBT234", self.root, image_loader)

        # Keep a list of all classes
        self.sequence_list = sorted(glob.glob(os.path.join(self.root, "*")))

        self.sequence_imgpath_list = []
        for seq_path in self.sequence_list:
            img_v_list = sorted(glob.glob(os.path.join(seq_path, "visible", "*")))
            img_i_list = sorted(glob.glob(os.path.join(seq_path, "infrared", "*")))
            self.sequence_imgpath_list.append(list(zip(img_v_list, img_i_list)))  # 存放索引对应各自模态文件名(因为文件名不一定是按bbox索引对应的....)

        # print(self.sequence_imgpath_list)
        # print(len(self.sequence_imgpath_list))
        # exit()
        self.sequence_info_list = [self._get_sequence_info_(seq_path) for seq_path in self.sequence_list]

    def _get_sequence_info_(self, seq_path):
        bbox = self._read_bb_anno_(seq_path)  # [v_tensor, i_tensor]
        valid = (bbox[:, 0, 2] > 0) & (bbox[:, 0, 3] > 0) & (bbox[:, 1, 2] > 0) & (bbox[:, 1, 3] > 0)
        visible = valid.clone().byte()
        return {"bbox": bbox, "valid": valid, "visible": visible}

    def get_name(self):
        return "RGBT234"

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno_(self, seq_path):
        gt_v = pandas.read_csv(
            os.path.join(seq_path, "visible.txt"), delimiter=",", header=None, dtype=np.float32, na_filter=True, low_memory=False
        ).values
        gt_i = pandas.read_csv(
            os.path.join(seq_path, "infrared.txt"), delimiter=",", header=None, dtype=np.float32, na_filter=True, low_memory=False
        ).values
        gt_vi = torch.cat([torch.tensor(gt_v).unsqueeze(1), torch.tensor(gt_i).unsqueeze(1)], dim=1)  # (N, 2, 4) xywh
        return gt_vi

    def get_sequence_info(self, seq_id):
        return self.sequence_info_list[seq_id]

    def _get_frame(self, seq_id, frame_id):
        img_v_path, img_i_path = self.sequence_imgpath_list[seq_id][frame_id]
        return [self.image_loader(img_v_path), self.image_loader(img_i_path)]

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
