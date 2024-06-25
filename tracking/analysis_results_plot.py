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
# parser.add_argument("--tracker_param", type=str, help="Name of config file.")
parser.add_argument("--dataset_name", type=str, help="数据集, 用来加载真实标签")
parser.add_argument("--name", type=str, help="加载结果的文件夹名, 路径在setting的测试结果的路径下.")
parser.add_argument("--report_name", type=str, help="保存文件夹名, 路径在setting的保存路径下.")

args = parser.parse_args()

dataset_name = args.dataset_name

# 出此下策, 原代码结构是要以tracker_param去找config文件, 并且结果路径都以这玩意开头
# 算了直接全遍历吧
env = env_settings()
rgbt_result_path = glob.glob(os.path.join(env.results_path, args.name, "*"))
rgbt_result_path = [path_ for path_ in rgbt_result_path if os.path.exists(os.path.join(path_, args.dataset_name))]
rgbt_suffix = [os.path.basename(temp) for temp in rgbt_result_path]
for suffix_ in rgbt_suffix:
    trackers.extend(
        trackerlist(
            name=args.name,  # 其实是保存路径
            parameter_name="",  # 置空即可
            dataset_name=args.dataset_name,
            run_ids=None,
            display_name=suffix_,
            save_path=suffix_,
        )
    )

dataset = get_dataset(dataset_name)

# plot_results(trackers, dataset, args.report_name, merge_results=False, plot_types=("success", "prec", "norm_prec"))
from lib.test.analysis.plot_results_cn import plot_results
plot_results(trackers, dataset, args.report_name, merge_results=False, plot_types=("success", "prec", "norm_prec"))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
