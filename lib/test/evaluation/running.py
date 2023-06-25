import numpy as np
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
import torch

from concurrent import futures
import multiprocessing as mp

import cv2
import subprocess


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)

    base_results_path_prefix = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    os.makedirs(os.path.split(base_results_path_prefix)[0], exist_ok=True)

    if tracker.params.vis_search == 1:
        video_save_path = os.path.join(tracker.results_dir, seq.dataset, "vis_video")
        os.makedirs(video_save_path, exist_ok=True)
        temp_video = os.path.join(video_save_path, "temp_{}.mp4".format(seq.name))  # 为了ffmpeg转换前的临时文件
        video_writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), 30, output["search_vis_image"][0].shape[1::-1])

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter="\t", fmt="%d")

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter="\t", fmt="%f")

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter="\t", fmt="%.2f")

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [
                        v,
                    ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == "search_vis_image":  # means tracker.params.vis_search == 1
            for img in data:
                video_writer.write(img)
            video_writer.release()
            subprocess.run(
                args=[
                    "ffmpeg",
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

        if key == "target_bbox":
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = "{}_{}.txt".format(base_results_path_prefix, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = "{}.txt".format(base_results_path_prefix)
                save_bb(bbox_file, data)

        if key == "all_boxes":
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = "{}_{}_all_boxes.txt".format(base_results_path_prefix, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = "{}_all_boxes.txt".format(base_results_path_prefix)
                save_bb(bbox_file, data)

        if key == "all_scores":
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = "{}_{}_all_scores.txt".format(base_results_path_prefix, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = "{}_all_scores.txt".format(base_results_path_prefix)
                save_score(bbox_file, data)

        elif key == "time":
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = "{}_{}_time.txt".format(base_results_path_prefix, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = "{}_time.txt".format(base_results_path_prefix)
                save_time(timings_file, data)


tracker_mp: Tracker = None  # 折腾了半天还是得用它


def init_worker(tracker: Tracker, debug=None):
    global tracker_mp
    idx = mp.current_process()._identity[0]  # from 1
    device_id = (idx - 1) % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    tracker.create_tracker(device_id, debug=debug)
    # print("init", id(tracker), mp.current_process().pid)
    tracker_mp = tracker


def run_sequence(seq: Sequence, debug=False):
    """Runs a tracker on a sequence."""
    """2021.1.2 Add multiple gpu support"""
    # try:
    #     # worker_name = multiprocessing.current_process().name
    #     # worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
    #     # gpu_id = worker_id % num_gpu
    #     # torch.cuda.set_device(gpu_id)
    # except:
    #     pass
    tracker: Tracker = tracker_mp
    # print("call", id(tracker_mp), mp.current_process().pid, id(tracker))

    def _results_exist():
        if seq.object_ids is None:
            bbox_file = os.path.join(tracker.results_dir, seq.dataset, "{}.txt".format(seq.name))
            return os.path.exists(bbox_file)
        else:
            bbox_files = ["{}/{}_{}.txt".format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print(
            "FPS: {}".format(-1),
            "Tracker: {} {} {} ,  Sequence: {}".format(tracker.name, tracker.training_parameter_name, tracker.run_id, seq.name),
        )
        return

    if debug:
        output = tracker.run_sequence(seq, debug=debug)
    else:
        # try:
        output = tracker.run_sequence(seq, debug=debug)
        # except Exception as e:
        #     print(e)
        #     return

    sys.stdout.flush()

    if isinstance(output["time"][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output["time"]])
        num_frames = len(output["time"])
    else:
        exec_time = sum(output["time"])
        num_frames = len(output["time"])

    print(
        "FPS: {:.2f}".format(num_frames / exec_time),
        "Tracker: {} {} {} ,  Sequence: {}".format(tracker.name, tracker.training_parameter_name, tracker.run_id, seq.name),
    )

    if not debug:
        _save_tracker_output(seq, tracker, output)


def run_dataset(dataset, trackers, debug=False, threads=0):
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

    if mode == "sequential":
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug)
    elif mode == "parallel":
        assert len(trackers) == 1, trackers  # 一个一个tracker来不行吗!!!
        with futures.ProcessPoolExecutor(
            max_workers=threads, mp_context=mp.get_context("spawn"), initializer=init_worker, initargs=(trackers[0], debug)
        ) as executor:
            fs = [executor.submit(run_sequence, seq, debug) for seq in dataset]
            for _, f in enumerate(futures.as_completed(fs)):
                if f.exception() is not None and f.exception() != 0:
                    print("[Error]", f.exception(), ', for the detail using "try f.result() + except"')
                    print(f.result())

    # elif mode == 'parallel':
    #     param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
    #     with multiprocessing.Pool(processes=threads) as pool:
    #         pool.starmap(run_sequence, param_list)
    print("Tracker({}) test in {} videos done!".format(trackers[0].name, len(dataset)))
