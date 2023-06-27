import numpy as np
from lib.test.evaluation.data import RGBT_Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import glob
import os


class RGBT234Dataset(BaseDataset):
    """ """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.rgbt234_path
        self.sequence_name_list = [os.path.basename(path_) for path_ in sorted(glob.glob(os.path.join(self.base_path, "*")))]

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_name_list])

    def _construct_sequence(self, video_name):
        frames_v = sorted(glob.glob(os.path.join(self.base_path, video_name, "visible", "*")))  # visible infrared
        frames_i = sorted(glob.glob(os.path.join(self.base_path, video_name, "infrared", "*")))

        anno_path_v = os.path.join(self.base_path, video_name, "visible.txt")  # x,y,w,h
        ground_truth_rect_v = load_text(anno_path_v, delimiter=(",", None), dtype=np.float64, backend="numpy")
        anno_path_i = os.path.join(self.base_path, video_name, "infrared.txt")  # x,y,w,h
        ground_truth_rect_i = load_text(anno_path_i, delimiter=(",", None), dtype=np.float64, backend="numpy")

        # visible好像不需要, eval的时候会处理
        return RGBT_Sequence(video_name, list(zip(frames_v, frames_i)), "RGBT234", list(zip(ground_truth_rect_v, ground_truth_rect_i)))

    def __len__(self):
        return len(self.sequence_name_list)
