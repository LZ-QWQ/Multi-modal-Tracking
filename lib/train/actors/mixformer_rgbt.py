from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.ce_utils import generate_mask_cond, adjust_keep_rate
import torch


class MixFormerRGBTActor(BaseActor):
    """Actor for training the TSP_online and TSP_cls_online"""

    def __init__(self, net, objective, loss_weight, settings, run_score_head=False, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.run_score_head = run_score_head
        self.cfg = cfg

    def train(self, mode=True):
        """
        重载掉base_actor
        """
        if self.run_score_head:
            self.net.eval()
            self.net.module.score_branch.train(mode)
        else:
            self.net.train(mode)
        # self.net.module.backbone_v.eval() # RGB固定

        # if self.settings.local_rank in [-1, 0]:
        #     print("[Warning], box_head using eval for fixed")
        # self.net.module.box_head.eval()  # DDP model

    def __call__(self, data):
        """
        args:
            由于dataloader没改collect_fn, 所以默认下貌似是这样的(倒也挺好), 与dataset或者说sampler关系极大
            data:
            {
                "template_images_*": N_t*[(NCHW)], []表列表
                "search_images_*": N_s*[(NCHW)],
                "template_anno_*": N_t*[(N,4)],
                "search_anno_*": N_s*[(N,4)],
                还有一些乱七八糟的
            }
            在 mixformer 的设定下 N_t 应该是2(且第一个是远的real template, 第二个online), N_s 应该是1

        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # 在这里移入GPU

        # forward pass
        template_image_v = data["template_images_v"][0].cuda(non_blocking=True)
        online_template_image_v = data["template_images_v"][1].cuda(non_blocking=True)

        template_image_i = data["template_images_i"][0].cuda(non_blocking=True)
        online_template_image_i = data["template_images_i"][1].cuda(non_blocking=True)

        search_image_v = data["search_images_v"][0].cuda(non_blocking=True)
        search_image_i = data["search_images_i"][0].cuda(non_blocking=True)

        search_bbox = data["search_anno_v"][0].cuda(non_blocking=True)  # 先用RGB做标签
        search_bboxes = box_xywh_to_xyxy(search_bbox)

        # print("net", self.net.device, template_image_v.device)
        if "CE_LOC" in self.cfg.MODEL.BACKBONE and self.cfg.MODEL.BACKBONE.CE_LOC:  # 存在且不为空列表
            box_mask_z = generate_mask_cond(
                self.cfg, template_image_v.shape[0], template_image_v.device, data["template_anno_v"][0]
            )  # 先用RGB做标签（过太久代码咋写的都快忘完了，依稀记得基本都用RGB标签。

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(
                data["epoch"],
                warmup_epochs=ce_start_epoch,
                total_epochs=ce_start_epoch + ce_warm_epoch,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0],
            )

            out_dict, _ = self.net(
                [template_image_v, template_image_i],
                [online_template_image_v, online_template_image_i],
                [search_image_v, search_image_i],
                run_score_head=self.run_score_head,
                gt_bboxes=search_bboxes,
                ce_template_mask=box_mask_z,
                ce_keep_rate=ce_keep_rate,
            )
        else:
            out_dict, _ = self.net(
                [template_image_v, template_image_i],
                [online_template_image_v, online_template_image_i],
                [search_image_v, search_image_i],
                run_score_head=self.run_score_head,
                gt_bboxes=search_bboxes,
            )

        # process the groundtruth
        gt_bboxes = search_bbox  # (Ns, batch, 4) (x1,y1,w,h)

        labels = None
        if "pred_scores" in out_dict:
            try:
                labels = data["label"].view(-1).cuda(non_blocking=True)  # (batch, ) 0 or 1
            except:
                raise Exception("Please setting proper labels for score branch.")

        # compute losses
        loss, status = self.compute_losses(out_dict, gt_bboxes, labels=labels)

        return loss, status

    def forward_pass(self, data, run_score_head):
        search_bboxes = box_xywh_to_xyxy(data["search_anno_v"][0].clone())  # 先用RGB做标签
        out_dict, _ = self.net(
            [data["template_images_v"][0], data["template_images_i"][0]],
            [data["template_images_v"][1], data["template_images_i"][1]],
            [data["search_images_v"][0], data["search_images_i"][0]],
            run_score_head=run_score_head,
            gt_bboxes=search_bboxes,
        )
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def compute_losses(self, pred_dict, gt_bbox, return_status=True, labels=None):
        # Get boxes
        pred_boxes = pred_dict["pred_boxes"]
        # print(pred_boxes[0], gt_bbox[0])
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = (
            box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        )  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute ciou and iou
        try:
            ciou_loss, iou = self.objective["ciou"](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            ciou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective["l1"](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # weighted sum
        loss = self.loss_weight["ciou"] * ciou_loss + self.loss_weight["l1"] * l1_loss

        # compute cls loss if neccessary
        if "pred_scores" in pred_dict:
            score_loss = self.objective["score"](pred_dict["pred_scores"].view(-1), labels)
            loss = score_loss * self.loss_weight["score"]

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            if "pred_scores" in pred_dict:
                status = {"Loss/total": loss.item(), "Loss/scores": score_loss.item()}
                # status = {"Loss/total": loss.item(),
                #           "Loss/scores": score_loss.item(),
                #           "Loss/giou": ciou_loss.item(),
                #           "Loss/l1": l1_loss.item(),
                #           "IoU": mean_iou.item()}
            else:
                status = {"Loss/total": loss.item(), "Loss/ciou": ciou_loss.item(), "Loss/l1": l1_loss.item(), "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
