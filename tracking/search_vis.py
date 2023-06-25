import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), "..")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset


import numpy as np
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
from lib.test.evaluation.mulit_trackers_multi_modal_vis import Vis_Trackers
import torch

from concurrent import futures
import multiprocessing as mp

import cv2
import subprocess
from typing import List


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: List[np.ndarray]):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)

    base_results_path_prefix = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    os.makedirs(os.path.split(base_results_path_prefix)[0], exist_ok=True)

    video_save_path = os.path.join(tracker.results_dir, seq.dataset, "vis_video")
    os.makedirs(video_save_path, exist_ok=True)
    temp_video = os.path.join(video_save_path, "temp_{}.mp4".format(seq.name))  # 为了ffmpeg转换前的临时文件
    video_writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), 5, output[0].shape[1::-1])

    for img in output:
        video_writer.write(img)
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
            os.path.join(video_save_path, "{}.mp4".format(seq.name)),
        ],
        check=True,
    )
    os.remove(temp_video)


tracker_mp: Vis_Trackers = None  # 折腾了半天还是得用它


def init_worker(tracker: Vis_Trackers):
    global tracker_mp
    idx = mp.current_process()._identity[0]  # from 1
    device_id = (idx - 1) % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    tracker.create_tracker()
    # print("init", id(tracker), mp.current_process().pid)
    tracker_mp = tracker


def run_sequence(seq: Sequence, tracker = None):
    if tracker is None:
        tracker: Tracker = tracker_mp

    def _results_exist():
        bbox_file = os.path.join(tracker.results_dir, seq.dataset, "{}.txt".format(seq.name))
        return os.path.exists(bbox_file)

    if _results_exist():
        print(
            "skip -- Tracker: {} {} ,  Sequence: {}".format(tracker.name, tracker.training_parameter_name, seq.name),
        )
        return

    output = tracker.run_sequence(seq)

    sys.stdout.flush()

    print(
        "Tracker: {} {} ,  Sequence: {}".format(tracker.name, tracker.training_parameter_name, seq.name),
    )

    _save_tracker_output(seq, tracker, output)


def run_dataset(dataset, trackers, threads=0):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    # multiprocessing.set_start_method('spawn', force=True)

    print("Evaluating {:4d} trackers on {:5d} sequences".format(len(trackers), len(dataset)))

    # multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
         mode = "sequential"
    else:
        mode = "parallel"

    if mode == "parallel":
        with futures.ProcessPoolExecutor(
            max_workers=threads, mp_context=mp.get_context("spawn"), initializer=init_worker, initargs=(trackers, )
        ) as executor:
            fs = [executor.submit(run_sequence, seq) for seq in dataset]
            for _, f in enumerate(futures.as_completed(fs)):
                if f.exception() is not None and f.exception() != 0:
                    print("[Error]", f.exception(), ', for the detail using "try f.result() + except"')
                    print(f.result())
    elif mode == "sequential":
        trackers.create_tracker()
        for seq in dataset:
            run_sequence(seq, trackers)
    else:
        raise NotImplementedError
    
    print("Tracker({}) test in {} videos done!".format(trackers.name, len(dataset)))


def main():
    parser = argparse.ArgumentParser(description="Run multi-modal multi-trackers on sequence or dataset for visualization.")
    parser.add_argument("--tracker_name", type=str, nargs="+", help="Name of tracking method.")
    parser.add_argument("--tracker_param", type=str, nargs="+", help="Name of config file, using to load model parameters.")
    parser.add_argument("--model", type=str, nargs="+", help="Tracking model path.")

    parser.add_argument("--dataset_name", type=str, default="otb", help="Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).")
    parser.add_argument("--sequence", type=str, default=[], nargs="*", help="Sequence name list.")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads.")

    # parser.add_argument("--params__model", type=str, default=None, help="Tracking model path.")
    parser.add_argument("--params__update_interval", type=int, default=None, help="Update interval of online tracking.")
    parser.add_argument("--params__online_sizes", type=int, default=None)
    parser.add_argument("--params__search_area_scale", type=float, default=None)
    parser.add_argument("--params__max_score_decay", type=float, default=1.0)
    parser.add_argument("--params__vis_attn", type=int, choices=[0, 1], default=0, help="Whether visualize the attention maps.")

    # TODO: 具体实现待完善，该参数计划主要用于tracker返回search特征
    parser.add_argument("--params__vis_search", type=int, choices=[0, 1], default=0, help="using for visualizing the search features.")

    parser.add_argument("--type", type=str, choices=["Unimodal", "RGBT"], default="RGBT", help="")
    parser.add_argument("--save_name_suffix", type=str, default="", help="接在 tracker_name 和 tracker_param 后用来区分保存路径名")

    # 如果下面这两存在, params__model 失效
    parser.add_argument("--checkpoint_dir", type=str, default="", help="要测试的训练结果的路径")
    # parser.add_argument("--checkpoint_name", type=str, default="", help="要测试的训练的名字(暂时是experiment 里的文件名(****.yaml),不包括.yaml)")
    args = parser.parse_args()

    tracker_params = {}
    for param in list(filter(lambda s: s.split("__")[0] == "params" and getattr(args, s) != None, args.__dir__())):
        tracker_params[param.split("__")[1]] = getattr(args, param)

    
    assert len(args.tracker_name) == len(args.tracker_param)
    assert len(args.tracker_param) == len(args.model)

    print(tracker_params)

    dataset = get_dataset(args.dataset_name)

    if args.sequence:  # 列表非空
        dataset = [dataset[args.sequence]]
    trackers = Vis_Trackers(
        args.tracker_name, args.tracker_param, args.model, args.dataset_name, tracker_params, save_suffix=args.save_name_suffix
    )

    run_dataset(dataset, trackers, args.threads)


if __name__ == "__main__":
    tic_ = cv2.getTickCount()
    main()
    second_all = round((cv2.getTickCount() - tic_) / cv2.getTickFrequency())
    second_ = second_all % 60
    minute_ = (second_all % 3600) // 60
    hour_ = second_all // 3600
    print("All time: {:02d}:{:02d}:{:02d} (H:M:S)".format(hour_, minute_, second_))
