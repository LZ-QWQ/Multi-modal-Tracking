import os
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import torch
import random
from collections import OrderedDict
from lib.train.admin import env_settings
import glob
import xml.etree.ElementTree as ET
import numpy as np


class LLVIPseq(BaseVideoDataset):
    """
    The LLVIP dataset. LLVIP is an image dataset for detection. Thus, we treat each image as a sequence of length 1.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", version="2014"):
        """ """
        root = env_settings().llvip_dir if root is None else root
        super().__init__("LLVIP", root, image_loader)

        image_path_list_v = sorted(
            glob.glob(os.path.join(self.root, "visible", "*", "*")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        image_path_list_i = sorted(
            glob.glob(os.path.join(self.root, "infrared", "*", "*")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        image_path_list_vi = list(zip(image_path_list_v, image_path_list_i))
        anno_list = sorted(glob.glob(os.path.join(self.root, "Annotations", "*")))
        self.sequence_imgpath_list = []
        self.sequence_info_list = []
        self._construct_everything_(image_path_list_vi, anno_list)

    def _construct_everything_(self, image_path_list_vi, anno_list):
        for image_vi, anno in zip(image_path_list_vi, anno_list):
            assert os.path.splitext(os.path.basename(image_vi[0]))[0] == os.path.splitext(os.path.basename(image_vi[1]))[0]
            assert os.path.splitext(os.path.basename(image_vi[0]))[0] == os.path.splitext(os.path.basename(anno))[0]
            xmltree = ET.parse(anno)
            objects = xmltree.findall("object")
            for object_ in objects:
                bndbox = object_.find("bndbox")
                bbox = [
                    int(bndbox.find("xmin").text),
                    int(bndbox.find("ymin").text),
                    int(bndbox.find("xmax").text),
                    int(bndbox.find("ymax").text),
                ]
                bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # xywh
                self.sequence_imgpath_list.append(image_vi)

                bbox = np.array([bbox])
                bbox = np.repeat(bbox[:, np.newaxis, :], 2, 1)  # N,4 to N,2,4
                bbox = torch.from_numpy(bbox)  # xywh

                valid = (bbox[:, 0, 2] > 0) & (bbox[:, 0, 3] > 0) & (bbox[:, 1, 2] > 0) & (bbox[:, 1, 3] > 0)
                visible = valid.clone().byte()
                self.sequence_info_list.append({"bbox": bbox, "valid": valid, "visible": visible})

    def is_video_sequence(self):
        return False  # 重要!!!

    def get_name(self):
        return "LLVIP"

    def get_num_sequences(self):
        return len(self.sequence_info_list)

    def get_sequence_info(self, seq_id):
        return self.sequence_info_list[seq_id]
    
    def _get_frame(self, seq_id):
        img_v_path, img_i_path = self.sequence_imgpath_list[seq_id]
        return [self.image_loader(img_v_path), self.image_loader(img_i_path)]

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # LLVIP is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frame(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...] for _ in frame_ids]

        object_meta = OrderedDict()
        return frame_list, anno_frames, object_meta


if __name__ == "__main__":
    import cv2

    test = LLVIPseq(image_loader=cv2.imread)
    print(test.get_num_sequences())
    # print(test.get_frames(223, [5, 1]))
