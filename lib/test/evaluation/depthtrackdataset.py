import numpy as np
from lib.test.evaluation.data import RGBT_Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import glob
import os


class DepthTrackDataset(BaseDataset):
    """
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.depthtrack_path
        self.sequence_path_list = sorted(glob.glob(os.path.join(self.base_path, "test", "*", "*")))

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_path_list])

    def _construct_sequence(self, video_path):
        frames_v = sorted(glob.glob(os.path.join(video_path, "color", "*")))
        frames_i = sorted(glob.glob(os.path.join(video_path, "depth", "*")))

        anno_path_v = os.path.join(video_path, "groundtruth.txt")  # x,y,w,h
        ground_truth_rect_v = load_text(anno_path_v, delimiter=(",", None), dtype=np.float64, backend="numpy")
        ground_truth_rect_v = np.nan_to_num(ground_truth_rect_v)
        # anno_path_i = os.path.join(video_path, "groundtruth.txt")  # x,y,w,h
        # ground_truth_rect_i = load_text(anno_path_i, delimiter=(",", None), dtype=np.float64, backend="numpy")
        ground_truth_rect_i = ground_truth_rect_v.copy()
        
        # visible好像不需要, eval的时候会处理
        return RGBT_Sequence(
            os.path.basename(video_path), list(zip(frames_v, frames_i)), "DepthTrack", list(zip(ground_truth_rect_v, ground_truth_rect_i))
        )

    def __len__(self):
        return len(self.sequence_path_list)
