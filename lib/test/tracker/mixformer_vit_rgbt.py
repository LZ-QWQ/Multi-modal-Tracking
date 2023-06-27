from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target

# for debug
import cv2
import os
from lib.models.mixformer_vit_rgbt import build_mixformer_vit_rgbt
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box


class MixFormer(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MixFormer, self).__init__(params)
        network = build_mixformer_vit_rgbt(params.cfg, train=False)
        missing_keys, unexpected_keys = network.load_state_dict(torch.load(self.params.checkpoint, map_location="cpu")["net"], strict=True)
        print("Load pretrained backbone checkpoint from:", self.params.checkpoint)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        print("Loading pretrained ViT done.")
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
        print("Update interval is: ", self.update_intervals)

    def initialize(self, image, info: dict):
        """
        image: [image_v, image_i]
        """
        # forward the template once, info["init_bbox"] : (bbox_v, bbox_i)
        z_patch_arr_v, _, _ = sample_target(
            image[0], info["init_bbox"][0], self.params.template_factor, output_sz=self.params.template_size
        )
        z_patch_arr_i, _, _ = sample_target(
            image[1], info["init_bbox"][0], self.params.template_factor, output_sz=self.params.template_size
        )
        template_v = self.preprocessor.process(z_patch_arr_v)
        template_i = self.preprocessor.process(z_patch_arr_i)

        self.template = [template_v, template_i]

        self.online_template = [template_v, template_i]
        # save states
        self.state = info["init_bbox"][0]  # simple using RGB bbox
        self.frame_id = 0
        if self.save_all_boxes:
            """save all predicted boxes"""
            all_boxes_save = info["init_bbox"] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
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

        search_v = self.preprocessor.process(x_patch_arr_v)
        search_i = self.preprocessor.process(x_patch_arr_i)

        with torch.no_grad():
            out_dict, _ = self.network(self.template, self.online_template, [search_v, search_i])

        pred_boxes = out_dict["pred_boxes"].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update template
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0:
                z_patch_arr_v, _, _ = sample_target(
                    image[0], self.state, self.params.template_factor, output_sz=self.params.template_size
                )  # (x1, y1, w, h)
                z_patch_arr_i, _, _ = sample_target(
                    image[1], self.state, self.params.template_factor, output_sz=self.params.template_size
                )  # (x1, y1, w, h)
                self.online_template = [self.preprocessor.process(z_patch_arr_v), self.preprocessor.process(z_patch_arr_i)]

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            """save all predictions"""
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state, "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return MixFormer
