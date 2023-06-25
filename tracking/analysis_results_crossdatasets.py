import _init_paths
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [8, 8]

from lib.test.analysis.plot_results import get_auc_pr
from lib.test.analysis.tracker import trackerlist
from lib.test.evaluation import get_dataset
import argparse

from lib.test.evaluation.environment import env_settings
import glob
import os
import torch
import numpy as np

env = env_settings()

parser = argparse.ArgumentParser(description="Run tracker on sequence or datasets_name.")
parser.add_argument("--tracker_param", type=str, help="Name of config file.")
parser.add_argument("--datasets_name", type=str, nargs="+", help="Name of config file.")
# parser.add_argument('--run_ids', type=str, help='Name of config file.')
# parser.add_argument('--run_ids', nargs='+', help='<Required> Set flag', required=True)
args = parser.parse_args()

# for all dataset
results_dict_dict = {}
# 保留baseline的结果做对比
baseline_name_path_list = [
    ("mixformer_vit_online", "baseline_large"),
    ("mixformer_vit_online", "baseline"),
    ("mixformer_vit", "baseline_large"),
    ("mixformer_vit", "baseline"),
]
for dataset_name in args.datasets_name:
    trackers = []
    for name, path in baseline_name_path_list:
        if os.path.exists(os.path.join(env.results_path, name, path, dataset_name)):
            trackers.extend(
                trackerlist(name=name, parameter_name="", dataset_name=dataset_name, run_ids=None, display_name=name + path, save_path=path)
            )

    # 出此下策, 原代码结构是要以tracker_param去找config文件, 并且结果路径都以这玩意开头
    rgbt_result_path = glob.glob(os.path.join(env.results_path, "mixformer_vit_rgbt", "*"))
    rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, dataset_name))]
    rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
    for suffix_ in rgbt_suffix:
        if "lasher" in suffix_:
            trackers.extend(
                trackerlist(
                    name="mixformer_vit_rgbt",
                    parameter_name=args.tracker_param,
                    dataset_name=dataset_name,
                    run_ids=None,
                    display_name="Unshared_" + suffix_,
                    save_path=suffix_,
                )
            )
    # shared
    env = env_settings()
    rgbt_result_path = glob.glob(os.path.join(env.results_path, "mixformer_vit_rgbt_shared", "*"))
    rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, dataset_name))]
    rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
    for suffix_ in rgbt_suffix:
        if "lasher" in suffix_:
            trackers.extend(
                trackerlist(
                    name="mixformer_vit_rgbt_shared",
                    parameter_name="",
                    dataset_name=dataset_name,
                    run_ids=None,
                    display_name="Shared_" + suffix_,
                    save_path=suffix_,
                )
            )

    # asymmetric_shared
    env = env_settings()
    rgbt_result_path = glob.glob(os.path.join(env.results_path, "asymmetric_shared", "*"))
    rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, dataset_name))]
    rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
    for suffix_ in rgbt_suffix:
        if "lasher" in suffix_:
            trackers.extend(
                trackerlist(
                    name="asymmetric_shared",
                    parameter_name="",
                    dataset_name=dataset_name,
                    run_ids=None,
                    display_name="Asym_Shared_" + suffix_,
                    save_path=suffix_,
                )
            )

    env = env_settings()
    rgbt_result_path = glob.glob(os.path.join(env.results_path, "asymmetric_shared_online", "*"))
    rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, dataset_name))]
    rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
    for suffix_ in rgbt_suffix:
        if "lasher" in suffix_:
            trackers.extend(
                trackerlist(
                    name="asymmetric_shared_online",
                    parameter_name="",
                    dataset_name=dataset_name,
                    run_ids=None,
                    display_name="Asym_Shared_Online" + suffix_,
                    save_path=suffix_,
                )
            )

    env = env_settings()
    rgbt_result_path = glob.glob(os.path.join(env.results_path, "mixformer_vit_rgbt_unibackbone", "*"))
    rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, dataset_name))]
    rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
    for suffix_ in rgbt_suffix:
        if "lasher" and "tracking" in suffix_:
            trackers.extend(
                trackerlist(
                    name="mixformer_vit_rgbt_unibackbone",
                    parameter_name="",
                    dataset_name=dataset_name,
                    run_ids=None,
                    display_name="Unibackbone" + suffix_,
                    save_path=suffix_,
                )
            )

    dataset = get_dataset(dataset_name)
    results_dict_dict[dataset_name] = get_auc_pr(trackers, dataset, dataset_name, merge_results=False)

tracker_set = []  # 用set排除可能没有可能有的结果
for dataset_name, result in results_dict_dict.items():
    tracker_set.extend(list(result.keys()))
tracker_set = np.array(list(set(tracker_set)))

temp_pad_result_auc = [
    results_dict_dict[args.datasets_name[0]][tracker] if tracker in results_dict_dict[args.datasets_name[0]].keys() else (-1.0, -1.0)
    for tracker in tracker_set
]
auc_pr = np.array(temp_pad_result_auc, dtype=[("auc", "f8"), ("pr", "f8")])
index_sorted = np.argsort(auc_pr, order=["auc", "pr"])[::-1]

tracker_set = tracker_set[index_sorted]
scores = {}
for dataset_name, result in results_dict_dict.items():
    scores[dataset_name] = [result[tracker] if tracker in result.keys() else (-1.0, -1.0) for tracker in tracker_set]


def generate_formatted_report(row_labels, scores, table_name=""):
    name_width = max([len(d) for d in row_labels] + [len(table_name)]) + 5
    min_score_width = 11

    report_text = "\n{label: <{width}} |".format(label=table_name, width=name_width)

    score_widths = [max(min_score_width, len(k) + 3) for k in scores.keys()]

    for s, s_w in zip(scores.keys(), score_widths):
        report_text = "{prev} {s: <{width}} |".format(prev=report_text, s=s, width=s_w)

    report_text = "{prev}\n".format(prev=report_text)

    for trk_id, d_name in enumerate(row_labels):
        # display name
        report_text = "{prev}{tracker: <{width}} |".format(prev=report_text, tracker=d_name, width=name_width)
        for (score_type, score_value), s_w in zip(scores.items(), score_widths):
            report_text = "{prev} {score: <{width}} |".format(
                prev=report_text, score="{:0.2f}/{:0.2f}".format(*score_value[trk_id]), width=s_w
            )
        report_text = "{prev}\n".format(prev=report_text)

    return report_text


report = generate_formatted_report(tracker_set, scores, "All Result AUC/PR.")
print(report)
