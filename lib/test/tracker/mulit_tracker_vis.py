import torch
from lib.train.data.processing_utils import sample_target

# for debug
import cv2
import os
from lib.models.mixformer_vit_rgbt.asymmetric_shared_ce import build_asymmetric_shared_ce
from lib.models.mixformer_vit_rgbt.asymmetric_shared import build_asymmetric_shared
from lib.models.mixformer_vit_rgbt.asymmetric_shared_online import build_asymmetric_shared_online_score
from lib.models.mixformer_vit_rgbt.mixformer import build_mixformer_vit_rgbt  # backbone 独立
from lib.models.mixformer_vit_rgbt.mixformer_unibackbone import build_mixformer_vit_rgbt_uni  # backbone 和 LN 用相同的
from lib.models.mixformer_vit_rgbt.mixformer_shared import build_mixformer_vit_rgbt_shared  # backbone 独立 + LN specific

from lib.test.tracker.tracker_utils import Preprocessor_Multimodal, vis_search
from lib.utils.box_ops import clip_box

from typing import List
from lib.test.utils import TrackerParams

import numpy as np

script_dict = {
    "asymmetric_shared_ce": build_asymmetric_shared_ce,
    "asymmetric_shared": build_asymmetric_shared,
    "asymmetric_shared_online": build_asymmetric_shared_online_score,
    "mixformer_vit_rgbt_shared": build_mixformer_vit_rgbt_shared,
    "mixformer_vit_rgbt": build_mixformer_vit_rgbt,
    "mixformer_vit_rgbt_unibackbone": build_mixformer_vit_rgbt_uni,
}


class Multi_Trackers:
    def __init__(self, name_list: List[str], params_list: List[TrackerParams], dataset_name):
        self.network_list = []
        self.params_list = params_list
        self.trackers_num = len(name_list)
        for name, params in zip(name_list, params_list):
            network = script_dict[name](params.cfg, train=False)
            device = torch.cuda.current_device()
            print(name, params.checkpoint)
            missing_keys, unexpected_keys = network.load_state_dict(
                torch.load(params.checkpoint, map_location=lambda storage, loc: storage.cuda(device))["net"], strict=True
            )
            self.network = network.cuda(device)
            self.network.eval()

            print(name, "model load pretrained backbone checkpoint from:", params.checkpoint)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained ViT done.")
            self.network_list.append(network)

        # self.cfg = params.cfg
        self.preprocessor = Preprocessor_Multimodal()
        self.state = None
        self.frame_id = 0

        # Set the update interval
        self.update_intervals_list = []
        for params in self.params_list:
            if hasattr(params.cfg.TEST.UPDATE_INTERVALS, dataset_name):
                self.update_intervals_list.append(params.cfg.TEST.UPDATE_INTERVALS[dataset_name])
            else:
                self.update_intervals_list.append(params.cfg.DATA.MAX_SAMPLE_INTERVAL)
        print("Update interval for all trackers are: ", self.update_intervals_list)

        # ========= tracking 超参数应当是一样的，为了复用代码直接取第一个 =========
        self.template_factor = self.params_list[0].template_factor
        self.template_size = self.params_list[0].template_size
        self.search_factor = self.params_list[0].search_factor
        self.search_size = self.params_list[0].search_size

        self.search_scale_factor = self.params_list[0].cfg.TEST.SEARCH_SCALE_JITTER
        self.search_offset_factor = self.params_list[0].cfg.TEST.SEARCH_CENTER_JITTER
        self.template_scale_factor = self.params_list[0].cfg.TEST.TEMPLATE_SCALE_JITTER
        self.template_offset_factor = self.params_list[0].cfg.TEST.TEMPLATE_CENTER_JITTER
        # ========================================================================

    def jitter_bbox(self, bbox, scale_factor, offset_factor):
        """
        jitter for better visualization
        """
        jitter_scale = np.exp(np.random.randn(2) * scale_factor)
        jitter_offset_factor = np.random.rand(2) - 0.5

        jittered_size = bbox[2:4] * jitter_scale
        max_offset = np.sqrt(jittered_size.prod()) * offset_factor
        jittered_center = bbox[0:2] + 0.5 * bbox[2:4] + jitter_offset_factor * max_offset
        return np.concatenate((jittered_center - 0.5 * jittered_size, jittered_size), axis=0)

    def transform_image_to_crop(self, box_in, box_extract, resize_factor, crop_sz):
        box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]
        box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

        box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
        box_out_wh = box_in[2:4] * resize_factor

        box_out = np.concatenate((box_out_center - 0.5 * box_out_wh, box_out_wh))
        return box_out / crop_sz[0]

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def initialize(self, image, info: dict):
        """
        image: [image_v, image_i]
        """
        # forward the template once, info["init_bbox"] : (bbox_v, bbox_i)
        z_patch_arr_v, _, _ = sample_target(image[0], info["init_bbox"][0], self.template_factor, output_sz=self.template_size)
        z_patch_arr_i, _, _ = sample_target(image[1], info["init_bbox"][0], self.template_factor, output_sz=self.template_size)

        template_list = self.preprocessor.process(z_patch_arr_v, z_patch_arr_i)

        self.template = template_list

        # for different trackers using different online template
        self.online_template_list = [template_list] * self.trackers_num

        # save states
        self.state = info["init_bbox"][0]  # simple using RGB bbox
        self.frame_id = 0

    def track(self, image, info: dict = None):
        """
        image: image_v, image_i
        return: only the vis image with cropped image, feature image, pred bbox, and real bbox
        """
        H, W, _ = image[0].shape

        real_bbox_ = np.array(info["init_bbox"][0])  # simple using RGB bbox and reset bbox to label
        self.state = self.jitter_bbox(
            real_bbox_, scale_factor=self.search_scale_factor, offset_factor=self.search_offset_factor
        )  # jitter the label

        self.frame_id += 1
        # print("frame id: {}".format(self.frame_id))
        x_patch_arr_v, resize_factor, _ = sample_target(
            image[0], self.state, self.search_factor, output_sz=self.search_size
        )  # (x1, y1, w, h)
        x_patch_arr_i, resize_factor, _ = sample_target(
            image[1], self.state, self.search_factor, output_sz=self.search_size
        )  # (x1, y1, w, h)

        real_bbox = self.transform_image_to_crop(
            real_bbox_, self.state, resize_factor, np.array([self.search_size, self.search_size])
        )  # (x1, y1, w, h)

        search_list = self.preprocessor.process(x_patch_arr_v, x_patch_arr_i)

        search_vif_list = []
        pred_bbox_list = []
        with torch.inference_mode():
            for network, online_template in zip(self.network_list, self.online_template_list):
                out_dict, _, search_feat_v, search_feat_i, search_fusion = network(
                    self.template, online_template, search_list, return_features=True
                )
                search_vif_list.append(
                    (
                        search_feat_v.cpu().numpy().squeeze(0).transpose(1, 2, 0),
                        search_feat_i.cpu().numpy().squeeze(0).transpose(1, 2, 0),
                        search_fusion.cpu().numpy().squeeze(0).transpose(1, 2, 0),
                    )
                )
                pred_boxes = out_dict["pred_boxes"].view(-1, 4)

                # for vis
                pred_box = pred_boxes.mean(dim=0).tolist()  # (cx, cy, w, h) [0,1]
                pred_box[0] = pred_box[0] - 0.5 * pred_box[2]
                pred_box[1] = pred_box[1] - 0.5 * pred_box[3]
                pred_bbox_list.append(pred_box)

                # for online update
                # pred_box_state = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                # self.state = clip_box(self.map_box_back(pred_box_state, resize_factor), H, W, margin=10)
                self.state = self.jitter_bbox(
                    real_bbox_, scale_factor=self.template_scale_factor, offset_factor=self.template_offset_factor
                )

        # update template
        for idx, info_ in enumerate(zip(self.update_intervals_list, self.online_template_list)):
            update_intervals, online_template = info_
            for update_i in update_intervals:
                if self.frame_id % update_i == 0:
                    z_patch_arr_v, _, _ = sample_target(
                        image[0], self.state, self.template_factor, output_sz=self.template_size
                    )  # (x1, y1, w, h)
                    z_patch_arr_i, _, _ = sample_target(
                        image[1], self.state, self.template_factor, output_sz=self.template_size
                    )  # (x1, y1, w, h)
                    self.online_template_list[idx] = self.preprocessor.process(z_patch_arr_v, z_patch_arr_i)
        vis_image = vis_search(x_patch_arr_v, x_patch_arr_i, search_vif_list, pred_bbox_list, real_bbox)
        return vis_image
