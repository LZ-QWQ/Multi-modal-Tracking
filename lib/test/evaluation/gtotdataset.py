import numpy as np
from lib.test.evaluation.data import RGBT_Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import glob
import os


class GTOTDataset(BaseDataset):
    """ """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.gtot_path
        self.sequence_name_list = self._get_sequence_name_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_name_list])

    def _construct_sequence(self, video_name):
        frames_v = sorted(glob.glob(os.path.join(self.base_path, video_name, "v", "*")))
        frames_i = sorted(glob.glob(os.path.join(self.base_path, video_name, "i", "*")))

        anno_path_v = os.path.join(self.base_path, video_name, "groundTruth_v.txt")  # x1,y1,x2,y2
        anno_path_i = os.path.join(self.base_path, video_name, "groundTruth_i.txt")  # x1,y1,x2,y2
        ground_truth_rect_v = load_text(anno_path_v, delimiter=(",", None), dtype=np.float64, backend="numpy")
        ground_truth_rect_i = load_text(anno_path_i, delimiter=(",", None), dtype=np.float64, backend="numpy")

        # to x1,y1,x2,y2
        ground_truth_rect_v[:, 2] = ground_truth_rect_v[:, 2] - ground_truth_rect_v[:, 0]  # w
        ground_truth_rect_v[:, 3] = ground_truth_rect_v[:, 3] - ground_truth_rect_v[:, 1]  # h

        ground_truth_rect_i[:, 2] = ground_truth_rect_i[:, 2] - ground_truth_rect_i[:, 0]  # w
        ground_truth_rect_i[:, 3] = ground_truth_rect_i[:, 3] - ground_truth_rect_i[:, 1]  # h

        return RGBT_Sequence(video_name, list(zip(frames_v, frames_i)), "GTOT", list(zip(ground_truth_rect_v, ground_truth_rect_i)))

    def _get_sequence_name_list(self):
        with open(os.path.join(self.base_path, "gtot.txt"), "r") as f:
            videos_name_list = f.readlines()
        videos_name_list = [temp.strip() for temp in videos_name_list]
        return videos_name_list

    def __len__(self):
        return len(self.sequence_name_list)
