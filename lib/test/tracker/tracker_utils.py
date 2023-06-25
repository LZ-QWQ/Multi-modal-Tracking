import torch
import numpy as np
from lib.utils.misc import NestedTensor
import matplotlib.pyplot as plt
import os
import cv2
from typing import List, Tuple


class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2, 0, 1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return NestedTensor(img_tensor_norm, amask_tensor)


class Preprocessor_wo_mask(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2, 0, 1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return img_tensor_norm


class Preprocessor_Multimodal(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_v: np.ndarray, img_i: np.ndarray):
        # Deal with the image patch
        img_i = cv2.applyColorMap(img_i, cv2.COLORMAP_JET)
        img_i = torch.tensor(img_i).cuda().float().permute((2, 0, 1)).unsqueeze(dim=0)
        img_v = torch.tensor(img_v).cuda().float().permute((2, 0, 1)).unsqueeze(dim=0)
        img_v = ((img_v / 255.0) - self.mean) / self.std  # (1,3,H,W)
        img_i = ((img_i / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return [img_v, img_i]


class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2, 0, 1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return img_tensor_norm, amask_tensor


class PreprocessorX_onnx(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std  # (1, 3, H, W)
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)


def vis_attn_maps(attn_weights, q_w, k_w, skip_len, x1, x2, x1_title, x2_title, save_path=".", idxs=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shape1 = [q_w, q_w]
    shape2 = [k_w, k_w]

    attn_weights_mean = []
    for attn in attn_weights:
        attn_weights_mean.append(attn[..., skip_len : (skip_len + k_w**2)].mean(dim=1).squeeze().reshape(shape1 + shape2).cpu())

    # downsampling factor
    fact = 32

    # let's select 4 reference points for visualization
    # idxs = [(32, 32), (64, 64), (32, 96), (96, 96), ]
    if idxs is None:
        idxs = [(64, 64)]

    block_num = 0
    idx_o = idxs[0]
    for attn_weight in attn_weights_mean:
        fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        ax = fig.add_subplot(111)
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(attn_weight[..., idx[0], idx[1]], cmap="cividis", interpolation="nearest")
        ax.axis("off")
        # ax.set_title(f'Stage2-Block{block_num}')
        plt.savefig(save_path + "/Stage2-Block{}_attn_weight.png".format(block_num))
        plt.close()
        block_num += 1

    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    x2_ax = fig.add_subplot(111)
    x2_ax.imshow(x2)
    x2_ax.axis("off")
    plt.savefig(save_path + "/{}.png".format(x2_title))
    plt.close()

    # the reference points as red circles
    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    x1_ax = fig.add_subplot(111)
    x1_ax.imshow(x1)
    for y, x in idxs:
        # scale = im.height / img.shape[-2]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        x1_ax.add_patch(plt.Circle((x, y), fact // 2, color="r"))
        # x1_ax.set_title(x1_title)
        x1_ax.axis("off")
    plt.savefig(save_path + "/{}.png".format(x1_title))
    plt.close()

    del attn_weights_mean


def vis_search(search_image_v, search_image_i, search_feature_vif_list: List[Tuple[np.ndarray, np.ndarray]], pred_bbox_list, real_bbox):
    """
    for visualization
    TODO: 把融合后特征也加入可视化
    TODO: 能不能把目标框给弄进来
    TODO: 第一帧怎么办？人麻了
    """
    color = [(247, 44, 200)[::-1], (44, 162, 247)[::-1], (239, 255, 66)[::-1], (2, 255, 250)[::-1]]

    out_sz = 160
    # 创建一个空白的图像，然后进行拼接
    search_vis_image = np.zeros((out_sz * 3, out_sz * (1 + len(search_feature_vif_list)), 3), dtype=np.uint8)

    # origin image
    search_image_v = cv2.resize(search_image_v, (out_sz, out_sz), interpolation=cv2.INTER_CUBIC)
    search_image_i = cv2.resize(search_image_i, (out_sz, out_sz), interpolation=cv2.INTER_CUBIC)
    search_image_v_origin = search_image_v.copy()
    search_image_i_origin = search_image_i.copy()

    real_bbox = (np.array(real_bbox) * out_sz).astype(np.int32)
    # 在图像上画出标注框
    # cv2.rectangle(search_image_v, (real_bbox[0], real_bbox[1]), (real_bbox[0] + real_bbox[2], real_bbox[1] + real_bbox[3]), (0, 255, 0), 2)
    # cv2.rectangle(search_image_i, (real_bbox[0], real_bbox[1]), (real_bbox[0] + real_bbox[2], real_bbox[1] + real_bbox[3]), (0, 255, 0), 2)

    for idx, search_vif in enumerate(search_feature_vif_list):
        search_feature_v, search_feature_i, search_fusion = search_vif

        # 《Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer》
        search_feature_v = _feature_func(search_feature_v, out_sz)
        search_feature_i = _feature_func(search_feature_i, out_sz)
        search_fusion = _feature_func(search_fusion, out_sz, p_norm=2)

        pred_bbox = pred_bbox_list[idx]
        pred_bbox = (np.array(pred_bbox) * out_sz).astype(np.int32)

        # cv2.rectangle(
        #     search_image_v, (pred_bbox[0], pred_bbox[1]), (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), color[idx], 2
        # )
        # cv2.rectangle(
        #     search_image_i, (pred_bbox[0], pred_bbox[1]), (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), color[idx], 2
        # )

        search_feature_v = (search_feature_v.astype(np.float32) + search_image_v_origin.astype(np.float32)) / 2.0
        search_feature_i = (search_feature_i.astype(np.float32) + search_image_i_origin.astype(np.float32)) / 2.0

        search_vis_image[: 1 * out_sz, out_sz * (idx + 1) : out_sz * (idx + 2)] = search_feature_v.astype(np.uint8)
        search_vis_image[1 * out_sz : 2 * out_sz, out_sz * (idx + 1) : out_sz * (idx + 2)] = search_feature_i.astype(np.uint8)
        search_vis_image[2 * out_sz :, out_sz * (idx + 1) : out_sz * (idx + 2)] = search_fusion.astype(np.uint8)

    search_vis_image[0:out_sz, 0:out_sz] = search_image_v
    search_vis_image[out_sz:2 * out_sz, 0:out_sz] = search_image_i

    # cv2.imshow("Debug", search_vis_image)
    # cv2.waitKey(0)
    return search_vis_image


def _feature_func(feature_map: torch.Tensor, out_sz, p_norm = 1):
    """
    vis_search 函数的一个辅助处理特征的函数
    """
    feature_map = np.mean(np.abs(feature_map)**p_norm, axis=2)
    feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map)) * 255
    feature_map = feature_map.astype(np.uint8)
    feature_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)
    feature_map = cv2.resize(feature_map, (out_sz, out_sz), interpolation=cv2.INTER_CUBIC)
    return feature_map
