import _init_paths
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results

# from lib.test.evaluation.tracker_rgbt import trackerlist
from lib.test.analysis.tracker import trackerlist
from lib.test.evaluation import get_dataset
import argparse

from lib.test.evaluation.environment import env_settings
import glob
import os

trackers = []

parser = argparse.ArgumentParser(description="Run tracker on sequence or dataset.")
parser.add_argument("--tracker_param", type=str, help="Name of config file.")
parser.add_argument("--dataset_name", type=str, help="Name of config file.")
# parser.add_argument('--run_ids', type=str, help='Name of config file.')
# parser.add_argument('--run_ids', nargs='+', help='<Required> Set flag', required=True)
args = parser.parse_args()

dataset_name = args.dataset_name

"""MixFormer"""
# trackers.extend(trackerlist(name='mixformer_online', parameter_name=args.tracker_param, dataset_name=args.dataset_name,
#                             run_ids=None, display_name='MixFormerOnline'))

if args.dataset_name == "RGBT234":
    trackers.extend(
        trackerlist(
            name="mixformer_vit_online",
            parameter_name="",
            dataset_name=args.dataset_name,
            run_ids=None,
            display_name="MixFormerOnline_Large",
            save_path="baseline_large",
        )
    )

    trackers.extend(
        trackerlist(
            name="mixformer_vit_online",
            parameter_name="",
            dataset_name=args.dataset_name,
            run_ids=None,
            display_name="MixFormerOnline_Large_TIR",
            save_path="baseline_large_TIR",
        )
    )

    trackers.extend(
        trackerlist(
            name="mixformer_vit_online",
            parameter_name="",
            dataset_name=args.dataset_name,
            run_ids=None,
            display_name="MixFormerOnline_Large_Prompt",
            save_path="baseline_large_Prompt",
        )
    )

# 出此下策, 原代码结构是要以tracker_param去找config文件, 并且结果路径都以这玩意开头
# 算了直接全遍历吧
env = env_settings()
rgbt_result_path = glob.glob(os.path.join(env.results_path, "mixformer_vit_rgbt", "*"))
rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, args.dataset_name))]
rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
for suffix_ in rgbt_suffix:
    if "lasher" and "tracking" in suffix_:
        trackers.extend(
            trackerlist(
                name="mixformer_vit_rgbt",
                parameter_name="",
                dataset_name=args.dataset_name,
                run_ids=None,
                display_name="Unshared_" + suffix_,
                save_path=suffix_,
            )
        )

env = env_settings()
rgbt_result_path = glob.glob(os.path.join(env.results_path, "mixformer_vit", "*"))
rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, args.dataset_name))]
rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
for suffix_ in rgbt_suffix:
    if "baseline_large_tir" in suffix_:
        trackers.extend(
            trackerlist(
                name="mixformer_vit",
                parameter_name="",
                dataset_name=args.dataset_name,
                run_ids=None,
                display_name="TIR_FN" + suffix_,
                save_path=suffix_,
            )
        )

# shared
# env = env_settings()
# rgbt_result_path = glob.glob(os.path.join(env.results_path, "mixformer_vit_rgbt_shared", "*"))
# rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, args.dataset_name))]
# rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
# for suffix_ in rgbt_suffix:
#     if "lasher" and "tracking" in suffix_:
#         trackers.extend(
#             trackerlist(
#                 name="mixformer_vit_rgbt_shared",
#                 parameter_name=args.tracker_param,
#                 dataset_name=args.dataset_name,
#                 run_ids=None,
#                 display_name="Shared_" + suffix_,
#                 save_path=suffix_,
#             )
#         )

env = env_settings()
rgbt_result_path = glob.glob(os.path.join(env.results_path, "asymmetric_shared", "*"))
rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, args.dataset_name))]
rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
for suffix_ in rgbt_suffix:
    if "lasher" and "tracking" in suffix_:
        trackers.extend(
            trackerlist(
                name="asymmetric_shared",
                parameter_name=args.tracker_param,
                dataset_name=args.dataset_name,
                run_ids=None,
                display_name="Asym_Shared_" + suffix_,
                save_path=suffix_,
            )
        )

env = env_settings()
rgbt_result_path = glob.glob(os.path.join(env.results_path, "asymmetric_shared_online", "*"))
rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, args.dataset_name))]
rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
for suffix_ in rgbt_suffix:
    if "lasher" and "tracking" in suffix_:
        trackers.extend(
            trackerlist(
                name="asymmetric_shared_online",
                parameter_name=args.tracker_param,
                dataset_name=args.dataset_name,
                run_ids=None,
                display_name="Asym_Shared_Online" + suffix_,
                save_path=suffix_,
            )
        )

env = env_settings()
rgbt_result_path = glob.glob(os.path.join(env.results_path, "mixformer_vit_rgbt_unibackbone", "*"))
rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, args.dataset_name))]
rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
for suffix_ in rgbt_suffix:
    if "lasher" and "tracking" in suffix_:
        trackers.extend(
            trackerlist(
                name="mixformer_vit_rgbt_unibackbone",
                parameter_name=args.tracker_param,
                dataset_name=args.dataset_name,
                run_ids=None,
                display_name="Unibackbone" + suffix_,
                save_path=suffix_,
            )
        )

# env = env_settings()
# rgbt_result_path = glob.glob(os.path.join(env.results_path, "asymmetric_shared_ce", "*"))
# rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, args.dataset_name))]
# rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
# for suffix_ in rgbt_suffix:
#     if "lasher" and "tracking" in suffix_:
#         trackers.extend(
#             trackerlist(
#                 name="asymmetric_shared_ce",
#                 parameter_name=args.tracker_param,
#                 dataset_name=args.dataset_name,
#                 run_ids=None,
#                 display_name="Asym_Shared_CE" + suffix_,
#                 save_path=suffix_,
#             )
#         )

dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=("success", "prec", "norm_prec"))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
