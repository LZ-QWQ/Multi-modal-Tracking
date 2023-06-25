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
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation import Tracker, RGBT_Tracker

import glob
import re


def run_tracker(
    tracker_name,
    tracker_param,
    run_id=None,
    dataset_name="otb",
    sequence=None,
    debug=0,
    threads=0,
    tracker_params=None,
    type=None,
    save_suffix="",
):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        type: RGBT or Unimodal
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    if type == "RGBT":
        trackers = [RGBT_Tracker(tracker_name, tracker_param, dataset_name, run_id, tracker_params=tracker_params, save_suffix=save_suffix)]
    elif type in ["RGB", "TIR", "Prompt"]:
        trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id, tracker_params=tracker_params, save_suffix=save_suffix, mode=type)]
    else:
        raise ValueError("type should be RGBT or Unimodal, now is {}".format(type))

    run_dataset(dataset, trackers, debug, threads)


def main():
    parser = argparse.ArgumentParser(description="Run tracker on sequence or dataset.")
    parser.add_argument("tracker_name", type=str, help="Name of tracking method.")
    parser.add_argument("tracker_param", type=str, help="Name of config file, using to load model parameters and checkpoint.")
    parser.add_argument("--runid", type=int, default=None, help="The run id.")
    parser.add_argument("--dataset_name", type=str, default="otb", help="Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).")
    parser.add_argument("--sequence", type=str, default=None, help="Sequence number or name.")
    parser.add_argument("--debug", type=int, default=0, help="Debug level.")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads.")
    parser.add_argument("--num_gpus", type=int, default=8)

    parser.add_argument("--params__model", type=str, default=None, help="Tracking model path.")
    parser.add_argument("--params__update_interval", type=int, default=None, help="Update interval of online tracking.")
    parser.add_argument("--params__online_sizes", type=int, default=None)
    parser.add_argument("--params__search_area_scale", type=float, default=None)
    parser.add_argument("--params__max_score_decay", type=float, default=1.0)
    parser.add_argument("--params__vis_attn", type=int, choices=[0, 1], default=0, help="Whether visualize the attention maps.")

    # TODO: 具体实现待完善，该参数计划主要用于tracker返回search特征
    parser.add_argument("--params__vis_search", type=int, choices=[0, 1], default=0, help="using for visualizing the search features.")

    parser.add_argument("--type", type=str, choices=["RGB", "TIR", "Prompt", "RGBT"], default="RGBT", help="")
    parser.add_argument("--save_name_suffix", type=str, default="", help="接在 tracker_name 和 tracker_param 后用来区分保存路径名")

    # 如果下面这两存在, params__model 失效
    parser.add_argument("--checkpoint_dir", type=str, default="", help="要测试的训练结果的路径")
    # parser.add_argument("--checkpoint_name", type=str, default="", help="要测试的训练的名字(暂时是experiment 里的文件名(****.yaml),不包括.yaml)")
    args = parser.parse_args()

    tracker_params = {}
    for param in list(filter(lambda s: s.split("__")[0] == "params" and getattr(args, s) != None, args.__dir__())):
        tracker_params[param.split("__")[1]] = getattr(args, param)

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    if args.checkpoint_dir != "":
        checkpoint_list = sorted(
            glob.glob(os.path.join(args.checkpoint_dir, "checkpoints", "train", args.tracker_name, args.tracker_param, "*"))
        ) 
        if "online" not in checkpoint_list[0] and "baseline_large_tir" not in checkpoint_list[0] :
            checkpoint_list = checkpoint_list[10:] # 非online的时候前10个抛弃
            
        print("Start test {} checkpoints".format(len(checkpoint_list)))
        for checkpoint in checkpoint_list:
            save_suffix = re.findall(r"ep\d*", checkpoint)[0]
            # 着实有点不知道咋办了, TODO 后面想办法把测试用参数命名 和 训练结果的参数命名分隔开
            save_suffix = "{}_{}_{}".format(save_suffix, args.tracker_param, args.save_name_suffix)
            tracker_params["model"] = checkpoint
            print(tracker_params)
            run_tracker(
                args.tracker_name,
                args.tracker_param,
                args.runid,
                args.dataset_name,
                seq_name,
                args.debug,
                args.threads,
                tracker_params=tracker_params,
                type=args.type,
                save_suffix=save_suffix,
            )

    else:
        print(tracker_params)
        run_tracker(
            args.tracker_name,
            args.tracker_param,
            args.runid,
            args.dataset_name,
            seq_name,
            args.debug,
            args.threads,
            tracker_params=tracker_params,
            type=args.type,
            save_suffix=args.save_name_suffix,
        )


if __name__ == "__main__":
    tic_ = cv2.getTickCount()
    main()
    second_all = round((cv2.getTickCount() - tic_) / cv2.getTickFrequency())
    second_ = second_all % 60
    minute_ = (second_all % 3600) // 60
    hour_ = second_all // 3600
    print("All time: {:02d}:{:02d}:{:02d} (H:M:S)".format(hour_, minute_, second_))
