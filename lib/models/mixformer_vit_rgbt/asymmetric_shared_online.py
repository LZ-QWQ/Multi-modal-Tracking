from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from einops import rearrange
from timm.models.layers import DropPath, Mlp

from lib.utils.misc import is_main_process
from lib.models.mixformer_cvt.head import build_box_head
from lib.models.mixformer_cvt.utils import to_2tuple
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.models.mixformer_vit_rgbt.pos_utils import get_2d_sincos_pos_embed
from lib.models.mixformer_vit_rgbt.fusion_utils import *
from lib.models.mixformer_cvt.score_decoder import ScoreDecoder


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        """
        Cross Modal Asymmetric Attention
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x_v, x_i, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x_v.shape
        qkv = self.qkv(torch.cat([x_v, x_i], dim=0)).reshape(B * 2, N, 3, self.num_heads, C // self.num_heads)
        qkv_V, qkv_I = torch.split(qkv, [B, B], dim=0)
        qkv_V = qkv_V.permute(2, 0, 3, 1, 4)
        qkv_I = qkv_I.permute(2, 0, 3, 1, 4)
        qV, kV, vV = qkv_V.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        qI, kI, vI = qkv_I.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q_mt_V, q_s_V = torch.split(qV, [t_h * t_w * 2, s_h * s_w], dim=2)
        k_mt_V, k_s_V = torch.split(kV, [t_h * t_w * 2, s_h * s_w], dim=2)
        v_mt_V, v_s_V = torch.split(vV, [t_h * t_w * 2, s_h * s_w], dim=2)

        q_mt_I, q_s_I = torch.split(qI, [t_h * t_w * 2, s_h * s_w], dim=2)
        k_mt_I, k_s_I = torch.split(kI, [t_h * t_w * 2, s_h * s_w], dim=2)
        v_mt_I, v_s_I = torch.split(vI, [t_h * t_w * 2, s_h * s_w], dim=2)

        k_mt = torch.cat([k_mt_V, k_mt_I], dim=2)  # t_h * t_w * 2 * 2
        v_mt = torch.cat([v_mt_V, v_mt_I], dim=2)

        #### asymmetric mixed attention ####
        # RGB templates to templates
        attn = (q_mt_V @ k_mt_V.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt_V = (attn @ v_mt_V).transpose(1, 2).reshape(B, t_h * t_w * 2, C)
        # TIR templates to templates
        attn = (q_mt_I @ k_mt_I.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt_I = (attn @ v_mt_I).transpose(1, 2).reshape(B, t_h * t_w * 2, C)
        # RGB search to self and all templates
        attn = (q_s_V @ torch.cat([k_mt, k_s_V], dim=2).transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s_V = (attn @ torch.cat([v_mt, v_s_V], dim=2)).transpose(1, 2).reshape(B, s_h * s_w, C)
        # TIR search to self and all templates
        attn = (q_s_I @ torch.cat([k_mt, k_s_I], dim=2).transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s_I = (attn @ torch.cat([v_mt, v_s_I], dim=2)).transpose(1, 2).reshape(B, s_h * s_w, C)

        x_V = torch.cat([x_mt_V, x_s_V], dim=1)
        x_I = torch.cat([x_mt_I, x_s_I], dim=1)
        x = self.proj_drop(self.proj(torch.cat([x_V, x_I], dim=0)))
        x_V, x_I = torch.split(x, [B, B], dim=0)
        return x_V, x_I


class Block_Shared(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        # 对不同模态采用相同参数和不同LN
        # assert isinstance(norm_layer, nn.LayerNorm), norm_layer

        super().__init__()
        self.norm1_v = norm_layer(dim)
        self.norm1_i = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2_v = norm_layer(dim)
        self.norm2_i = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_v, x_i, t_h, t_w, s_h, s_w):
        B, N, C = x_v.shape
        # print(x_v.shape, "喵喵喵")
        x_vi = torch.cat([x_v, x_i], dim=0)  # residual

        x_v = self.norm1_v(x_v)
        x_i = self.norm1_i(x_i)
        x_v, x_i = self.attn(x_v, x_i, t_h, t_w, s_h, s_w)
        x = torch.cat([x_v, x_i], dim=0)
        x_vi = x_vi + self.drop_path1(x)
        x_v, x_i = torch.split(x_vi, [B, B], dim=0)

        x_v = self.norm2_v(x_v)
        x_i = self.norm2_i(x_i)
        x = torch.cat([x_v, x_i], dim=0)
        x_vi = x_vi + self.drop_path2(self.mlp(x))
        x_v, x_i = torch.split(x_vi, [B, B], dim=0)
        return x_v, x_i


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        img_size_s=256,
        img_size_t=128,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
    ):
        super(VisionTransformer, self).__init__(
            img_size=224,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        self.patch_embed = embed_layer(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block_Shared(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.grid_size_s = img_size_s // patch_size
        self.grid_size_t = img_size_t // patch_size
        self.num_patches_s = self.grid_size_s**2
        self.num_patches_t = self.grid_size_t**2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim), requires_grad=False)

        self.init_pos_embed()

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], int(self.num_patches_t**0.5), cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], int(self.num_patches_s**0.5), cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, x_t, x_ot, x_s):
        """
        堆叠RGBT, cat([batch_v,batch_i])
        :param x_t: (batch * 2, c, 128, 128) x_ot
        :param x_s: (batch * 2, c, 288, 288)
        :return:
        """
        x_t = self.patch_embed(x_t)  # BCHW-->BNC
        x_ot = self.patch_embed(x_ot)
        x_s = self.patch_embed(x_s)
        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.grid_size_s
        H_t = W_t = self.grid_size_t

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x = torch.cat([x_t, x_ot, x_s], dim=1)
        x = self.pos_drop(x)

        x_v, x_i = torch.split(x, [B // 2, B // 2], dim=0)
        for blk in self.blocks:
            x_v, x_i = blk(x_v, x_i, H_t, W_t, H_s, W_s)
        x = torch.cat([x_v, x_i], dim=0)
        x_t, x_ot, x_s = torch.split(x, [H_t * W_t, H_t * W_t, H_s * W_s], dim=1)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        return x_t_2d, x_ot_2d, x_s_2d


def get_mixformer_vit(config, train):
    img_size_s = config.DATA.SEARCH.SIZE
    img_size_t = config.DATA.TEMPLATE.SIZE
    if config.MODEL.VIT_TYPE == "large_patch16":
        vit = VisionTransformer(
            img_size_s=img_size_s,
            img_size_t=img_size_t,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=0.1,
        )
    elif config.MODEL.VIT_TYPE == "base_patch16":
        vit = VisionTransformer(
            img_size_s=img_size_s,
            img_size_t=img_size_t,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=0.1,
        )
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'large_patch16' or 'base_patch16'")

    # 消掉该死的没用的ViT的参数
    vit.cls_token = None
    vit.pos_embed = None
    vit.norm = None
    vit.head = None

    if config.MODEL.BACKBONE.PRETRAINED and train:
        ckpt_path = config.MODEL.BACKBONE.PRETRAINED_PATH
        device = torch.cuda.current_device()
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))["model"]
        new_dict = {}
        for k, v in ckpt.items():
            if "pos_embed" not in k and "mask_token" not in k:  # use fixed pos embed
                if "norm1" in k:
                    k_v = k.replace("norm1", "norm1_v")
                    k_i = k.replace("norm1", "norm1_i")
                    new_dict[k_v] = v
                    new_dict[k_i] = v
                elif "norm2" in k:
                    k_v = k.replace("norm2", "norm2_v")
                    k_i = k.replace("norm2", "norm2_i")
                    new_dict[k_v] = v
                    new_dict[k_i] = v
                else:
                    new_dict[k] = v
        missing_keys, unexpected_keys = vit.load_state_dict(new_dict, strict=False)
        if is_main_process():
            print("Load pretrained backbone checkpoint from:", ckpt_path)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained ViT done.")

    return vit


class MixFormer_RGBT_OnlineScore(nn.Module):
    def __init__(self, backbone, box_head, fusion_vi, score_branch, head_type="CORNER"):
        """
        Initializes the model.
        backbone: [backbone_v, backbone_i]
        """
        super().__init__()
        self.backbone = backbone

        self.fusion_vi = fusion_vi
        self.box_head = box_head
        self.head_type = head_type
        self.score_branch = score_branch

    def forward(self, template, online_template, search, run_score_head=False, gt_bboxes=None, return_features=False):
        """
        template: [v,i]
        online_template: [v,i]
        search: [v,i]
        """
        template = torch.cat([template[0], template[1]], dim=0)
        online_template = torch.cat([online_template[0], online_template[1]], dim=0)
        search = torch.cat([search[0], search[1]], dim=0)
        template, online_template, search = self.backbone(template, online_template, search)

        N = search.size(0) // 2  # B*2, 768, 18, 18
        search_v, search_i = torch.split(search, [N, N], dim=0)
        search = self.fusion_vi(search_v.contiguous(), search_i.contiguous())  # B, C, H, W
        # search shape: (b, 384, 20, 20)
        # Forward the corner head
        # simply using RGB template, due to the implictly fusion in backbone
        if return_features:
            # 2024.03.31 修改
            # return *self.forward_head(search, torch.split(template, [N, N], dim=0)[0], run_score_head), search_v, search_i, search
            return *self.forward_head(search, torch.cat(torch.split(template, [N, N], dim=0), dim=2), run_score_head), search_v, search_i, search
        else:
            # return self.forward_head(search, torch.split(template, [N, N], dim=0)[0], run_score_head)
            return self.forward_head(search, torch.cat(torch.split(template, [N, N], dim=0), dim=2), run_score_head)

    def forward_test(self, search, run_score_head=True, gt_bboxes=None):
        # search: (b, c, h, w)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, search = self.backbone.forward_test(search)
        # search (b, 384, 20, 20)
        # Forward the corner head
        return self.forward_box_head(search)

    def set_online(self, template, online_template):
        """
        template: [v, i]
        online_template: [v, i]
        search: [v, i]
        """
        # if template.dim() == 5:
        #     template = template.squeeze(0)
        # if online_template.dim() == 5:
        #     online_template = online_template.squeeze(0)
        self.backbone_v.set_online(template[0], online_template[0])
        self.backbone_i.set_online(template[1], online_template[1])

    def forward_head(self, search, template, run_score_head=False, gt_bboxes=None):
        """
        :param search: (b, c, h, w)
        :return:
        """
        out_dict = {}
        out_dict_box, outputs_coord = self.forward_box_head(search)
        out_dict.update(out_dict_box)
        if run_score_head:
            # forward the classification head
            if gt_bboxes is None:
                gt_bboxes = box_cxcywh_to_xyxy(outputs_coord.clone().view(-1, 4))
            # (b,c,h,w) --> (b,h,w)
            out_dict.update({"pred_scores": self.score_branch(search, template, gt_bboxes).view(-1)})
        return out_dict, outputs_coord

    def forward_box_head(self, search):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "CORNER" in self.head_type:
            # run the corner head
            b = search.size(0)
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(search))
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out_dict = {"pred_boxes": outputs_coord_new}
            return out_dict, outputs_coord_new
        else:
            raise KeyError


def build_asymmetric_shared_online_score(cfg, train=True) -> MixFormer_RGBT_OnlineScore:
    backbone = get_mixformer_vit(cfg, train)  # backbone without positional encoding and attention mask

    # fusion_vi = RGBT_Fusion_1(768 * 2, 768)  # 768对应 base_patch16
    # fusion_vi = DeformableAttentionFusion(768, num_feature_levels=2)
    # fusion_vi = Attention_Fusion_512(768, d_model=512, num_feature_levels=2)
    # fusion_vi = Attention_Fusion_Bimodal(768, d_model=512, num_feature_levels=2, num_encoder_layers=cfg.MODEL.FUSION_LAYERS)

    # fusion_vi = Attention_Fusion_Bimodal_LNSpecific(768, d_model=512, num_feature_levels=2, num_encoder_layers=cfg.MODEL.FUSION_LAYERS)

    fusion_vi = globals()[cfg.MODEL.FUSION_CLASS](768, d_model=512, num_feature_levels=2, num_encoder_layers=cfg.MODEL.FUSION_LAYERS)

    score_branch = ScoreDecoder(pool_size=4, hidden_dim=cfg.MODEL.HIDDEN_DIM, num_heads=cfg.MODEL.HIDDEN_DIM // 64)

    box_head = build_box_head(cfg)  # a simple corner head

    model = MixFormer_RGBT_OnlineScore(backbone, box_head, fusion_vi, score_branch, head_type=cfg.MODEL.HEAD_TYPE)

    # 先加载SPM，后面不会覆盖到前面的
    if cfg.MODEL.SCORE_PRETRAINED_PATH != "" and train:
        ckpt_path = cfg.MODEL.SCORE_PRETRAINED_PATH
        device = torch.cuda.current_device()
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))["net"]
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        if is_main_process():
            print("Load pretrained checkpoint from:", ckpt_path)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained SPM done.")
            
    if cfg.MODEL.TRACKER_PRETRAINED_PATH != "" and train:
        ckpt_path = cfg.MODEL.TRACKER_PRETRAINED_PATH
        device = torch.cuda.current_device()
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))["net"]
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        if is_main_process():
            print("Load pretrained checkpoint from:", ckpt_path)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained tracker done.")
    else:
        if train:
            print("[INFO] not load pretrain model, only using MAE initilization")



    return model
