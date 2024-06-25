import os
import glob
import time

import torch
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from lib.train.dataset.depth_utils import get_rgbd_frame

from lib.test.analysis.tracker import trackerlist, RGBT_Tracker

from typing import List

from lib.test.evaluation import get_dataset
from lib.test.evaluation.environment import env_settings

import cv2
import argparse
from lib.test.utils.load_text import load_text
import subprocess

from concurrent import futures
import multiprocessing as mp

# 这个文件用于将用video_demo文件在H20T-RGB上跟踪出的结果转至H20T-TIR上
class RGBT_Vis_Tracker:
    """
    伪tracker, 读图读标签画video
    """

    def __init__(
        self,
        dataset_name: str,
        trackers: List[RGBT_Tracker],
        report_name: str,
    ):
        # assert run_id is None or isinstance(run_id, int)
        self.dataset = get_dataset(dataset_name)
        self.report_name = report_name
        self.trackers = trackers

        settings = env_settings()
        result_plot_path = os.path.join(settings.result_plot_path, report_name)
        self.video_save_path = os.path.join(result_plot_path, "vis_videos")  # using to save mp4
        os.makedirs(self.video_save_path, exist_ok=True)

        self.color = [(247, 44, 200)[::-1], (44, 162, 247)[::-1], (239, 255, 66)[::-1], (2, 255, 250)[::-1]]
        assert len(trackers) <= len(self.color)

    def run(self):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        # Get init information
        for seq in self.dataset:
            self._run_sequence(seq)

    def _run_sequence(self, seq):
        if os.path.exists(os.path.join(self.video_save_path, "{}.mp4".format(seq.name))):
            print("Video: {:12s} exist, skip".format(seq.name))
            return

        tic = cv2.getTickCount()
        pred_bb_list = []
        for trk_id, trk in enumerate(self.trackers):
            # Load results
            results_path = os.path.join(trk.results_dir, seq.dataset, "{}.txt".format(seq.name))
            if os.path.isfile(results_path):
                pred_bb = load_text(str(results_path), delimiter=("\t", ",", " "), dtype=np.float64)
                pred_bb_list.append(pred_bb)
            else:
                raise Exception("Result not found. {}".format(results_path))

        video_dataset = Video(seq.frames, seq.dataset in ["DepthTrack"])  # RGBD数据集要单独处理读取，（因为深度图稍微没做过预处理呢）
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

        anno_bb = np.array(seq.ground_truth_rect)[:, 0, :]  # 只取RGB
        valid = (anno_bb[:, 2:] > 0.0).sum(1) == 2
        for idx, image_vi in enumerate(video_loader):
            # image = image_vi[0]
            if idx == 0:
                temp_video = os.path.join(self.video_save_path, "temp_{}.mp4".format(seq.name))  # 为了ffmpeg转换前的临时文件
                video_writer = cv2.VideoWriter(
                    temp_video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (2 * image_vi[0].shape[1], image_vi[0].shape[0])
                )

            # 不画了！
            # if idx % 10 == 0:
            #     idx_sparse = idx // 10
            #     if valid[idx_sparse]:
            #         gt_bbox = anno_bb[idx_sparse].astype(np.int32)
            #         cv2.rectangle(image, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 0, 255), 2)

            for trk_id, pred_bb in enumerate(pred_bb_list):
                pred_bbox = pred_bb[idx].astype(np.int32)
                cv2.rectangle(
                    image_vi[0],
                    (pred_bbox[0], pred_bbox[1]),
                    (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]),
                    self.color[trk_id],
                    2,
                )
                cv2.rectangle(
                    image_vi[1],
                    (pred_bbox[0], pred_bbox[1]),
                    (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]),
                    self.color[trk_id],
                    2,
                )
            image = np.concatenate(image_vi, axis=1)
            # cv2.putText(image, str(idx), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
            video_writer.write(image)
            # if idx == 20:
            #     break

        video_writer.release()
        subprocess.run(
            args=[
                "/usr/bin/ffmpeg",
                "-nostdin",  # https://github.com/kkroening/ffmpeg-python/issues/108 解决回显关闭问题
                "-y",
                "-loglevel",
                "quiet",
                "-i",
                temp_video,
                "-vcodec",
                "h264",
                os.path.join(self.video_save_path, "{}.mp4".format(seq.name)),
            ],
            check=True,
        )
        os.remove(temp_video)
        toc = cv2.getTickCount() - tic
        toc /= cv2.getTickFrequency()
        print("Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps".format(seq.name, toc, len(seq.frames) / toc))


class Video(Dataset):
    def __init__(self, frame_vi_list, isdepth=False) -> None:
        super().__init__()
        self.frame_vi_list = frame_vi_list
        self.isdepth = isdepth

    def __len__(self):
        return len(self.frame_vi_list)

    def read_image(self, image_file):
        im = cv2.imread(image_file)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        if self.isdepth:  # 深度读取稍微不太一样
            image_v, image_i = get_rgbd_frame(self.frame_vi_list[index][0], self.frame_vi_list[index][1], dtype="rgb3d", depth_clip=True)
        else:
            image_v = self.read_image(self.frame_vi_list[index][0])
            image_i = self.read_image(self.frame_vi_list[index][1])
        return (image_v, image_i)


def collect_image_single(batch):
    return batch[0]


trackers = []


def get_args():
    parser = argparse.ArgumentParser(description="Run tracker on sequence or dataset.")
    parser.add_argument("--dataset_name", type=str, default="VTUAV", help="Name of dataset.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    trackers = []
    if args.dataset_name == "VTUAV":
        trackers.extend(
            trackerlist(
                name="asymmetric_shared_online",
                parameter_name="",
                dataset_name=args.dataset_name,
                run_ids=None,
                display_name="Ours",
                save_path="tracking",
            )
        )
    if trackers:
        run_tracker = RGBT_Vis_Tracker(args.dataset_name, trackers, args.dataset_name)
        with futures.ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context("spawn")) as executor:
            fs = [executor.submit(run_tracker._run_sequence, seq) for seq in run_tracker.dataset]
            for _, f in enumerate(futures.as_completed(fs)):
                if f.exception() is not None and f.exception() != 0:
                    print("[Error]", f.exception(), ', for the detail using "try f.results() + except"')
                    print(f.result())
            print("{} videos done!".format(len(fs)))
