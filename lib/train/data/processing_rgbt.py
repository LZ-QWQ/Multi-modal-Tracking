import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
import random
import numpy as np


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
    through the network. For example, it can be used to crop a search region around the object, apply various data
    augmentations, etc."""

    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {
            "template": transform if template_transform is None else template_transform,
            "search": transform if search_transform is None else search_transform,
            "joint": joint_transform,
        }

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class MixformerProcessing(BaseProcessing):
    """The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(
        self,
        search_area_factor,
        output_sz,
        center_jitter_factor,
        scale_jitter_factor,
        mode="pair",
        settings=None,
        train_score=False,
        *args,
        **kwargs
    ):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.train_score = train_score
        # self.label_function_params = label_function_params
        self.out_feat_sz = 20  ######## the output feature map size

    def _get_jittered_box(self, bbox_vi, mode):
        """Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        # 这里有个问题就是, 要不要针对RGB和TIR存在的弱不对齐问题做数据增强
        # 最简单的方法就是直接用RGB的框(有的工作用TIR), 先用这个, 简单也许已经够用
        # TODO 另一个计划是考虑TIR的框在RGB的基础上再做一点扰动(因为有双模态的标签, 可以先对齐再扰动来做数据增强)
        # 不过这个代码好像 crop的框和标签框是分开算的,,,改的时候再算
        jitter_scale = torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        jitter_offset_factor = torch.rand(2) - 0.5

        return [
            self._get_jittered_bbox_single_(bbox_temp, jitter_scale, jitter_offset_factor, self.center_jitter_factor[mode])
            for bbox_temp in bbox_vi
        ]

    def _get_jittered_bbox_single_(self, bbox, scale, offset_factor, max_offset_factor):
        jittered_size = bbox[2:4] * scale
        max_offset = jittered_size.prod().sqrt() * torch.tensor(max_offset_factor).float()
        jittered_center = bbox[0:2] + 0.5 * bbox[2:4] + max_offset * offset_factor
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_neg_proposals(self, box, min_iou=0.0, max_iou=0.3, sigma=0.5):
        """Generates proposals by adding noise to the input box
        args:
            box - input box
        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        # num_proposals = self.proposal_params['boxes_per_frame']
        # proposal_method = self.proposal_params.get('proposal_method', 'default')

        # if proposal_method == 'default':
        num_proposals = box.size(0)
        proposals = torch.zeros((num_proposals, 4)).to(box.device)
        gt_iou = torch.zeros(num_proposals)
        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box[i], min_iou=min_iou, max_iou=max_iou, sigma_factor=sigma)
        # elif proposal_method == 'gmm':
        #     proposals, _, _ = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
        #                                                                      num_samples=num_proposals)
        #     gt_iou = prutils.iou(box.view(1,4), proposals.view(-1,4))

        # # Map to [-1, 1]
        # gt_iou = gt_iou * 2 - 1
        return proposals

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform["joint"] is not None:
            data["template_images"], data["template_anno"] = self.transform["joint"](
                image=data["template_images"], bbox=data["template_anno"]
            )
            data["search_images"], data["search_anno"] = self.transform["joint"](
                image=data["search_images"], bbox=data["search_anno"], new_roll=False
            )

        for s in ["template", "search"]:
            assert self.mode == "sequence" or len(data[s + "_images"]) == 1, "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + "_anno"]]  # RGB-TIR

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            # 先只用RGB的框
            w, h = (
                torch.stack([bbox_vi[0] for bbox_vi in jittered_anno], dim=0)[:, 2],
                torch.stack([bbox_vi[0] for bbox_vi in jittered_anno], dim=0)[:, 3],
            )

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data["valid"] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops_v, boxes_v, att_mask_v, _ = prutils.jittered_center_crop(
                [images_vi[0] for images_vi in data[s + "_images"]],
                [bbox_vi[0] for bbox_vi in jittered_anno],
                [bbox_vi[0] for bbox_vi in data[s + "_anno"]],
                self.search_area_factor[s],
                self.output_sz[s],
            )
            crops_i, boxes_i, att_mask_i, _ = prutils.jittered_center_crop(
                [images_vi[1] for images_vi in data[s + "_images"]],
                [bbox_vi[0] for bbox_vi in jittered_anno],  # using RGB bbox
                [bbox_vi[1] for bbox_vi in data[s + "_anno"]],  # 先算出TIR标签, 用不用再说
                self.search_area_factor[s],
                self.output_sz[s],
            )

            # zip 产生元组......
            crops = list(zip(crops_v, crops_i))
            boxes = list(zip(boxes_v, boxes_i))
            att_mask = list(zip(att_mask_v, att_mask_i))

            # Apply transforms
            data[s + "_images"], data[s + "_anno"], data[s + "_att"] = self.transform[s](image=crops, bbox=boxes, att=att_mask, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele_vi in data[s + "_att"]:
                if (ele_vi[0] == 1).all():  # 虽然着实没太看懂这在干嘛, 但好像都用RGB的bbox做crop时应该是一样的 ba
                    data["valid"] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele_vi in data[s + "_att"]:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele_vi[0][None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data["valid"] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data["valid"] = True
        # Prepare output
        # if self.mode == "sequence":
        #     data = data.apply(stack_tensors)
        # else:
        #     data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        return data

    def _generate_regression_mask(self, target_center, mask_w, mask_h, mask_size=20):
        """
        NHW format
        :return:
        """
        k0 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, 1, -1)
        k1 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, -1, 1)

        d0 = (k0 - target_center[:, 0].view(-1, 1, 1)).abs()  # w, (b, 1, w)
        d1 = (k1 - target_center[:, 1].view(-1, 1, 1)).abs()  # h, (b, h, 1)
        # dist = d0.abs() + d1.abs()
        mask_w = mask_w.view(-1, 1, 1)
        mask_h = mask_h.view(-1, 1, 1)

        mask0 = torch.where(d0 <= mask_w * 0.5, torch.ones_like(d0), torch.zeros_like(d0))  # (b, 1, w)
        mask1 = torch.where(d1 <= mask_h * 0.5, torch.ones_like(d1), torch.zeros_like(d1))  # (b, h, 1)

        return mask0 * mask1  # (b, h, w)
