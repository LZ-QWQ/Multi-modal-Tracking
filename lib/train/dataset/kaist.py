import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import glob
import json


class KAIST(BaseVideoDataset):
    """
    这个数据集不是普通的单目标跟踪数据集, 单帧内有多个track
    为了适配原框架, 针对单个track_id做成一个序列
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
        self.root = env_settings().kaist_dir if root is None else root
        super().__init__("KAIST", self.root, image_loader)

        self.sequence_imgpath_list = []
        self.sequence_info_list = []
        self._construct_everything_()

    def _construct_everything_(self):
        video_list = sorted(glob.glob(os.path.join(self.root, "images", "*")))
        anno_list = sorted(glob.glob(os.path.join(self.root, "annotations", "*")))

        all_video_list = []
        all_anno_list = []
        for path, anno in zip(video_list, anno_list):
            assert os.path.basename(path) == os.path.basename(anno)
            temp_list_1 = sorted(glob.glob(os.path.join(path, "*")))
            temp_list_2 = sorted(glob.glob(os.path.join(anno, "*.json")))
            all_video_list += temp_list_1
            all_anno_list += temp_list_2

        for video_path, anno_path in zip(all_video_list, all_anno_list):
            with open(anno_path, encoding="UTF-8") as f:
                anno = json.load(f)
            image_list_v = sorted(glob.glob(os.path.join(video_path, "visible", "*")))
            image_list_i = sorted(glob.glob(os.path.join(video_path, "lwir", "*")))

            # python3.6 之后字典是有序的!!!!!
            for _, FrameIdx_bbox in anno.items():
                idx_ = np.array(list(FrameIdx_bbox.keys()), dtype=np.int32)
                imgpath_vi = list(zip(image_list_v[idx_[0] : idx_[-1] + 1], image_list_i[idx_[0] : idx_[-1] + 1]))
                self.sequence_imgpath_list.append(imgpath_vi)
                bbox = np.array(list(FrameIdx_bbox.values()))
                bbox = np.repeat(bbox[:, np.newaxis, :], 2, 1)  # N,4 to N,2,4
                bbox = torch.from_numpy(bbox)  # xywh

                valid = (bbox[:, 0, 2] > 0) & (bbox[:, 0, 3] > 0) & (bbox[:, 1, 2] > 0) & (bbox[:, 1, 3] > 0)
                visible = valid.clone().byte()
                self.sequence_info_list.append({"bbox": bbox, "valid": valid, "visible": visible})
                assert ((idx_[1:] - idx_[:-1]) == 1).all()  # 连续假设, 不连续就不好处理了
                assert len(bbox) == len(imgpath_vi)

    def get_name(self):
        return "KAIST"

    def get_num_sequences(self):
        return len(self.sequence_info_list)

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


if __name__ == "__main__":
    import cv2

    test = KAIST(image_loader=cv2.imread)
    print(test.get_num_sequences())
    # print(test.get_frames(223, [5]))
