from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pdb
import cv2
import torch

# import vot
import sys
import time
import os
from lib.test.evaluation.tracker_rgbt import RGBT_Tracker
import lib.test.vot_rgbd_test.vot as vot
from lib.test.vot_rgbd_test.vot22_utils import *
from lib.train.dataset.depth_utils import get_rgbd_frame

from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target

import cv2
import os
from lib.models.mixformer_vit_rgbt.asymmetric_shared_online import build_asymmetric_shared_online_score
from lib.test.tracker.tracker_utils import Preprocessor_Multimodal
from lib.utils.box_ops import clip_box
from lib.test.parameter.asymmetric_shared_online import parameters


class Tracer_VOT(BaseTracker):
    def __init__(self, params, dataset_name):
        super().__init__(params)
        network = build_asymmetric_shared_online_score(params.cfg, train=False)
        device = torch.cuda.current_device()
        missing_keys, unexpected_keys = network.load_state_dict(
            torch.load(self.params.checkpoint, map_location=lambda storage, loc: storage.cuda(device))["net"], strict=True
        )
        print("Load pretrained backbone checkpoint from:", self.params.checkpoint)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        print("Loading pretrained ViT done.")

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor_Multimodal()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0

        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
        print("Update interval is: ", self.update_intervals)

    def initialize(self, image, selection):
        """
        image: [image_v, image_i]
        """
        x, y, w, h = selection
        bbox = [x, y, w, h]
        # forward the template once, info["init_bbox"] : (bbox_v, bbox_i)
        z_patch_arr_v, _, _ = sample_target(image[0], bbox, self.params.template_factor, output_sz=self.params.template_size)
        z_patch_arr_i, _, _ = sample_target(image[1], bbox, self.params.template_factor, output_sz=self.params.template_size)

        template_list = self.preprocessor.process(z_patch_arr_v, z_patch_arr_i)

        self.template = template_list
        self.online_template = template_list
        self.max_pred_score = -1

        # save states
        self.state = bbox  # simple using RGB bbox
        self.frame_id = 0

    def track(self, image):
        """
        image: image_v, image_i
        """
        H, W, _ = image[0].shape
        self.frame_id += 1
        # print("frame id: {}".format(self.frame_id))
        x_patch_arr_v, resize_factor, _ = sample_target(
            image[0], self.state, self.params.search_factor, output_sz=self.params.search_size
        )  # (x1, y1, w, h)
        x_patch_arr_i, _, _ = sample_target(
            image[1], self.state, self.params.search_factor, output_sz=self.params.search_size
        )  # (x1, y1, w, h)

        search_list = self.preprocessor.process(x_patch_arr_v, x_patch_arr_i)

        with torch.inference_mode():
            out_dict, _ = self.network(self.template, self.online_template, search_list, run_score_head=True)

        pred_boxes = out_dict["pred_boxes"].view(-1, 4)
        pred_score = out_dict["pred_scores"].view(1).sigmoid().item()
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if pred_score > 0.5 and pred_score > self.max_pred_score:
            z_patch_arr_v, _, _ = sample_target(
                image[0], self.state, self.params.template_factor, output_sz=self.params.template_size
            )  # (x1, y1, w, h)
            z_patch_arr_i, _, _ = sample_target(
                image[1], self.state, self.params.template_factor, output_sz=self.params.template_size
            )  # (x1, y1, w, h)

            self.online_max_template = self.preprocessor.process(z_patch_arr_v, z_patch_arr_i)
            self.max_pred_score = pred_score

        # update template
        for _, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0:
                self.online_template = self.online_max_template
                self.online_max_template = self.template
                self.max_pred_score = -1

        return self.state, pred_score

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def run_vot_exp(
    training_parameter_name: str,
    tracker_params=None,
    vis=False,
    tracking_parameter_name: str = "tracking",
    out_conf=True,
    channel_type="rgbd",
):
    # ===================== deal with tracker_params =====================
    search_area_scale = None
    if tracker_params is not None and "search_area_scale" in tracker_params:
        search_area_scale = tracker_params["search_area_scale"]
    model = ""
    if tracker_params is not None and "model" in tracker_params:
        model = tracker_params["model"]
    params = parameters(training_parameter_name, tracking_parameter_name, model, search_area_scale)
    if tracker_params is not None:
        for param_k, v in tracker_params.items():
            setattr(params, param_k, v)

    tracker = Tracer_VOT(params=params, dataset_name="DepthTrack")

    if channel_type == "rgb":
        channel_type = None
    handle = vot.VOT("rectangle", channels=channel_type)

    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    print(imagefile)
    # read rgbd data
    if isinstance(imagefile, list) and len(imagefile) == 2:
        image = get_rgbd_frame(imagefile[0], imagefile[1], dtype="rgb3d", depth_clip=True)
    else:
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right

    tracker.initialize(image, selection)
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break

        # read rgbd data
        if isinstance(imagefile, list) and len(imagefile) == 2:
            image = get_rgbd_frame(imagefile[0], imagefile[1], dtype="rgb3d", depth_clip=True)
        else:
            image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right

        b1, max_score = tracker.track(image)

        if out_conf:
            handle.report(vot.Rectangle(*b1), max_score)
        else:
            handle.report(vot.Rectangle(*b1))
