import os
import sys
# env_Path不知道为什么根本没正常运行
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
sys.path.insert(0, '/home/lizheng/data4/MixFormer/lib/test/vot_rgbd_test')
from lib.test.vot_rgbd_test.tracker_class import run_vot_exp
import argparse

parser = argparse.ArgumentParser(description="Run tracker on sequence or dataset.")
parser.add_argument("tracker_param", type=str, help="Name of training config file, using to load model parameters and checkpoint.")

parser.add_argument("--threads", type=int, default=0, help="Number of threads.")
parser.add_argument("--num_gpus", type=int, default=8)

parser.add_argument("--params__model", type=str, default=None, help="Tracking model path.")
parser.add_argument("--params__update_interval", type=int, default=None, help="Update interval of online tracking.")
parser.add_argument("--params__online_sizes", type=int, default=None)
parser.add_argument("--params__search_area_scale", type=float, default=None)
parser.add_argument("--params__max_score_decay", type=float, default=1.0)

args = parser.parse_args()
tracker_params = {}
for param in list(filter(lambda s: s.split("__")[0] == "params" and getattr(args, s) != None, args.__dir__())):
    tracker_params[param.split("__")[1]] = getattr(args, param)

run_vot_exp(args.tracker_param, tracker_params,vis=False, out_conf=True, channel_type="rgbd")
