import numpy as np
from lib.test.evaluation.data import RGBT_Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import glob
import os


class VTUAVTrainDataset(BaseDataset):
    """ """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vtuav_path
        split_list = sorted(glob.glob(os.path.join(self.base_path, "train_data", "*")))
        self.sequence_path_list = []
        for path in split_list:
            self.sequence_path_list += sorted(glob.glob(os.path.join(path, "*")))

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_path_list])

    def _construct_sequence(self, video_path):
        frames_v = sorted(glob.glob(os.path.join(video_path, "rgb", "*")))
        frames_i = sorted(glob.glob(os.path.join(video_path, "ir", "*")))

        anno_path_v = os.path.join(video_path, "rgb.txt")  # x,y,w,h
        ground_truth_rect_v = load_text(anno_path_v, delimiter=(" ", None), dtype=np.float64, backend="numpy")
        anno_path_i = os.path.join(video_path, "ir.txt")  # x,y,w,h
        ground_truth_rect_i = load_text(anno_path_i, delimiter=(" ", None), dtype=np.float64, backend="numpy")

        # visible好像不需要, eval的时候会处理
        return RGBT_Sequence(os.path.basename(video_path), list(zip(frames_v, frames_i)), "VTUAV_TRAIN", list(zip(ground_truth_rect_v, ground_truth_rect_i)))

    def __len__(self):
        return len(self.sequence_path_list)
