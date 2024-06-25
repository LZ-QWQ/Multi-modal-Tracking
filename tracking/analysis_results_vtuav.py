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
parser.add_argument("--dataset_split", type=str, default="", help="Name of config file.") # 临时用于VTUAV的子集测试
# parser.add_argument('--run_ids', type=str, help='Name of config file.')
# parser.add_argument('--run_ids', nargs='+', help='<Required> Set flag', required=True)
args = parser.parse_args()

dataset_name = args.dataset_name + args.dataset_split

"""MixFormer"""
# trackers.extend(trackerlist(name='mixformer_online', parameter_name=args.tracker_param, dataset_name=args.dataset_name,
#                             run_ids=None, display_name='MixFormerOnline'))

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
    trackers.extend(
        trackerlist(
            name="mixformer_vit_online",
            parameter_name="",
            dataset_name=args.dataset_name,
            run_ids=None,
            display_name="RGB_Large",
            save_path="baseline_large",
        )
    )
    trackers.extend(
        trackerlist(
            name="mixformer_vit_online",
            parameter_name="",
            dataset_name=args.dataset_name,
            run_ids=None,
            display_name="RGB",
            save_path="baseline",
        )
    )
env = env_settings()
rgbt_result_path = glob.glob(os.path.join(env.results_path, "asymmetric_shared", "*"))
rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, args.dataset_name))]
rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
for suffix_ in rgbt_suffix:
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

rgbt_result_path = glob.glob(os.path.join(env.results_path, "asymmetric_shared_online", "*"))
rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, args.dataset_name))]
rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
for suffix_ in rgbt_suffix:
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



dataset = get_dataset(dataset_name)

print("start eval len(trackers)",len(trackers))
print_results(trackers, dataset, dataset_name , merge_results=False, plot_types=("success", "prec", "norm_prec"))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
