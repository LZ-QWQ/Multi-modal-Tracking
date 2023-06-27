import torch
from torch import nn
from mmcv.ops import ModulatedDeformConv2d, ModulatedDeformConv2dPack
from lib.models.mixformer_vit_rgbt.deformable_attention.deformable_encoder import DeformableAttentionFusion
from lib.models.mixformer_vit_rgbt.deformable_attention.deformable_encoder_lnspecific import DeformableAttentionFusion_LNSpecific


class RGBT_Fusion_1(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.fusion = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.fusion_relu = nn.ReLU(inplace=True)

        self.fusion2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.fusion2_bn = nn.BatchNorm2d(out_channels)
        self.fusion2_relu = nn.ReLU(inplace=True)

    def forward(self, input_v, input_i):
        out = self.fusion(torch.cat([input_v, input_i], dim=1))
        out = self.fusion_relu(self.fusion_bn(out))
        out = self.fusion2_relu(self.fusion2_bn(self.fusion2(out)))
        return out


class RGBT_Fusion_2(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        # deform_groups 2
        self.fusion_offset = nn.Conv2d(
            in_channels,
            2 * 3 * 3 * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )  # 差分还是cat呢 ? 先cat

        self.fusion = ModulatedDeformConv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, deform_groups=2, bias=False
        )
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.fusion_relu = nn.ReLU(inplace=True)

        self.fusion2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.fusion2_bn = nn.BatchNorm2d(out_channels)
        self.fusion2_relu = nn.ReLU(inplace=True)

        # offset initilization 0
        nn.init.zeros_(self.fusion_offset.weight)
        nn.init.zeros_(self.fusion_offset.bias)

    def forward(self, input_v, input_i):
        offset = self.fusion_offset(torch.cat([input_v, input_i], dim=1))
        o1, o2, mask = torch.chunk(offset, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        out = self.fusion(torch.cat([input_v, input_i], dim=1), offset, mask)
        out = self.fusion_relu(self.fusion_bn(out))
        out = self.fusion2_relu(self.fusion2_bn(self.fusion2(out)))
        return out


class RGBT_Fusion_3(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        # deform_groups 2
        self.fusion = ModulatedDeformConv2dPack(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, deform_groups=1, bias=False
        )
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.fusion_relu = nn.ReLU(inplace=True)

        self.fusion2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.fusion2_bn = nn.BatchNorm2d(out_channels)
        self.fusion2_relu = nn.ReLU(inplace=True)

    def forward(self, input_v, input_i):
        out = self.fusion(torch.cat([input_v, input_i], dim=1))
        out = self.fusion_relu(self.fusion_bn(out))
        out = self.fusion2_relu(self.fusion2_bn(self.fusion2(out)))
        return out


class RGBT_Fusion_Cat(nn.Module):
    # for ablation study
    def __init__(self, channels_num, **karwgs) -> None:
        # 只有第一个参数有用，karwgs接住乱七八糟的参数
        super().__init__()
        self.fusion1 = nn.Conv2d(
            in_channels=2 * channels_num, out_channels=2 * channels_num, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.fusion1_bn = nn.BatchNorm2d(2 * channels_num)
        self.fusion1_relu = nn.ReLU(inplace=True)

        self.fusion2 = nn.Conv2d(in_channels=2 * channels_num, out_channels=channels_num, kernel_size=3, stride=1, padding=1, bias=False)
        self.fusion2_bn = nn.BatchNorm2d(channels_num)
        self.fusion2_relu = nn.ReLU(inplace=True)

        self.fusion3 = nn.Conv2d(in_channels=channels_num, out_channels=channels_num, kernel_size=3, stride=1, padding=1, bias=False)
        self.fusion3_bn = nn.BatchNorm2d(channels_num)
        self.fusion3_relu = nn.ReLU(inplace=True)

    def forward(self, input_v, input_i):
        out = self.fusion1(torch.cat([input_v, input_i], dim=1))
        out = self.fusion1_relu(self.fusion1_bn(out))
        out = self.fusion2_relu(self.fusion2_bn(self.fusion2(out)))
        out = self.fusion3_relu(self.fusion3_bn(self.fusion3(out)))
        return out


class Attention_Fusion_1(nn.Module):
    def __init__(self, d_model, num_feature_levels=2) -> None:
        """ """
        super().__init__()
        self.fusion_attention = DeformableAttentionFusion(d_model, num_feature_levels=num_feature_levels)

    def forward(self, input_v, input_i):
        _, c, h, w = input_v.size()  # B, 768, 18, 18
        out = self.fusion_attention(input_v, input_i)  # B, 2*H*W, C
        out_v, out_i = torch.chunk(out, 2, 1)
        out = out_v + out_i  # 先加吧
        out = out.permute(0, 2, 1).view(-1, c, h, w).contiguous()
        return out


class Attention_Fusion_512(nn.Module):
    def __init__(self, channels_num, d_model=512, num_feature_levels=2) -> None:
        """
        in_channels 输入的两个模态的单个模态的通道数
        d_model attention的通道数
        输入假定不等于d_model, 768->d_model
        2*768 → 2*d_model → 768
        """
        super().__init__()
        self.adjust_v = nn.Sequential(
            nn.Conv2d(channels_num, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )
        self.adjust_i = nn.Sequential(
            nn.Conv2d(channels_num, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )

        self.fusion_attention = DeformableAttentionFusion(d_model, num_feature_levels=num_feature_levels, DeformAttn_type="normal")

        self.adjust_cat = nn.Sequential(
            nn.Conv2d(2 * d_model, channels_num, kernel_size=1),
            nn.GroupNorm(32, channels_num),
        )

    def forward(self, input_v, input_i):
        b, c, h, w = input_v.size()  # B, 768, 18, 18
        input_v = self.adjust_v(input_v)
        input_i = self.adjust_i(input_i)
        out = self.fusion_attention(input_v, input_i)  # B, 2*H*W, C
        out_v, out_i = torch.chunk(out, 2, 1)  # B, H*W, C
        out_v = out_v.permute(0, 2, 1).view(b, -1, h, w).contiguous()
        out_i = out_i.permute(0, 2, 1).view(b, -1, h, w).contiguous()
        out = self.adjust_cat(torch.cat([out_v, out_i], dim=1))
        return out


class Attention_Fusion_Bimodal(nn.Module):
    def __init__(self, channels_num, d_model=512, num_feature_levels=2, num_encoder_layers=6) -> None:
        """
        in_channels 输入的两个模态的单个模态的通道数
        d_model attention的通道数
        输入假定不等于d_model, 768->d_model
        2*768 → 2*d_model → 768
        """
        super().__init__()
        self.adjust_v = nn.Sequential(
            nn.Conv2d(channels_num, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )
        self.adjust_i = nn.Sequential(
            nn.Conv2d(channels_num, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )

        self.fusion_attention = DeformableAttentionFusion(
            d_model, num_encoder_layers=num_encoder_layers, num_feature_levels=num_feature_levels, DeformAttn_type="bimodal"
        )

        self.adjust_cat = nn.Sequential(
            nn.Conv2d(2 * d_model, channels_num, kernel_size=1),
            nn.GroupNorm(32, channels_num),
        )

    def forward(self, input_v, input_i):
        b, c, h, w = input_v.size()  # B, 768, 18, 18
        input_v = self.adjust_v(input_v)
        input_i = self.adjust_i(input_i)
        out = self.fusion_attention(input_v, input_i)  # B, 2*H*W, C
        out_v, out_i = torch.chunk(out, 2, 1)  # B, H*W, C
        out_v = out_v.permute(0, 2, 1).view(b, -1, h, w).contiguous()
        out_i = out_i.permute(0, 2, 1).view(b, -1, h, w).contiguous()
        out = self.adjust_cat(torch.cat([out_v, out_i], dim=1))
        return out


class Attention_Fusion_Bimodal_2(nn.Module):
    def __init__(self, channels_num, d_model=512, num_feature_levels=2, num_encoder_layers=6) -> None:
        """
        in_channels 输入的两个模态的单个模态的通道数
        d_model attention的通道数
        输入假定不等于d_model, 768->d_model
        2*768 → 2*d_model → 768
        """
        super().__init__()
        self.adjust_v = nn.Sequential(
            nn.Conv2d(channels_num, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )
        self.adjust_i = nn.Sequential(
            nn.Conv2d(channels_num, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )

        self.fusion_attention = DeformableAttentionFusion(
            d_model, num_encoder_layers=num_encoder_layers, num_feature_levels=num_feature_levels, DeformAttn_type="bimodal_2"
        )

        self.adjust_cat = nn.Sequential(
            nn.Conv2d(2 * d_model, channels_num, kernel_size=1),
            nn.GroupNorm(32, channels_num),
        )

    def forward(self, input_v, input_i):
        b, c, h, w = input_v.size()  # B, 768, 18, 18
        input_v = self.adjust_v(input_v)
        input_i = self.adjust_i(input_i)
        out = self.fusion_attention(input_v, input_i)  # B, 2*H*W, C
        out_v, out_i = torch.chunk(out, 2, 1)  # B, H*W, C
        out_v = out_v.permute(0, 2, 1).view(b, -1, h, w).contiguous()
        out_i = out_i.permute(0, 2, 1).view(b, -1, h, w).contiguous()
        out = self.adjust_cat(torch.cat([out_v, out_i], dim=1))
        return out


class Attention_Fusion_Bimodal_LNSpecific(nn.Module):
    def __init__(self, channels_num, d_model=512, num_feature_levels=2, num_encoder_layers=6) -> None:
        """
        in_channels 输入的两个模态的单个模态的通道数
        d_model attention的通道数
        输入假定不等于d_model, 768->d_model
        2*768 → 2*d_model → 768
        """
        super().__init__()
        self.adjust_v = nn.Sequential(
            nn.Conv2d(channels_num, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )
        self.adjust_i = nn.Sequential(
            nn.Conv2d(channels_num, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )

        self.fusion_attention = DeformableAttentionFusion_LNSpecific(
            d_model, num_encoder_layers=num_encoder_layers, num_feature_levels=num_feature_levels, DeformAttn_type="bimodal"
        )

        self.adjust_cat = nn.Sequential(
            nn.Conv2d(2 * d_model, channels_num, kernel_size=1),
            nn.GroupNorm(32, channels_num),
        )

    def forward(self, input_v, input_i):
        b, c, h, w = input_v.size()  # B, 768, 18, 18
        input_v = self.adjust_v(input_v)
        input_i = self.adjust_i(input_i)
        out = self.fusion_attention(input_v, input_i)  # B, 2*H*W, C
        out_v, out_i = torch.chunk(out, 2, 1)  # B, H*W, C
        out_v = out_v.permute(0, 2, 1).view(b, -1, h, w).contiguous()
        out_i = out_i.permute(0, 2, 1).view(b, -1, h, w).contiguous()
        out = self.adjust_cat(torch.cat([out_v, out_i], dim=1))
        return out


class Attention_Fusion_Bimodal_LNSpecific_Sum(nn.Module):
    def __init__(self, channels_num, d_model=512, num_feature_levels=2, num_encoder_layers=6) -> None:
        """
        in_channels 输入的两个模态的单个模态的通道数
        d_model attention的通道数
        输入假定不等于d_model, 768->d_model
        2*768 → 2*d_model → 768
        """
        super().__init__()
        self.adjust_v = nn.Sequential(
            nn.Conv2d(channels_num, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )
        self.adjust_i = nn.Sequential(
            nn.Conv2d(channels_num, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )

        self.fusion_attention = DeformableAttentionFusion_LNSpecific(
            d_model, num_encoder_layers=num_encoder_layers, num_feature_levels=num_feature_levels, DeformAttn_type="bimodal"
        )

        self.adjust_sum = nn.Sequential(
            nn.Conv2d(d_model, channels_num, kernel_size=1),
            nn.GroupNorm(32, channels_num),
        )

    def forward(self, input_v, input_i):
        b, c, h, w = input_v.size()  # B, 768, 18, 18
        input_v = self.adjust_v(input_v)
        input_i = self.adjust_i(input_i)
        out = self.fusion_attention(input_v, input_i)  # B, 2*H*W, C
        out_v, out_i = torch.chunk(out, 2, 1)  # B, H*W, C
        out = out_v + out_i
        out = out.permute(0, 2, 1).view(b, -1, h, w).contiguous()
        out = self.adjust_sum(out)
        return out


class Attention_Fusion_Bimodal_LNSpecific_2(nn.Module):
    def __init__(self, channels_num, d_model=512, num_feature_levels=2, num_encoder_layers=6) -> None:
        """
        in_channels 输入的两个模态的单个模态的通道数
        d_model attention的通道数
        输入假定不等于d_model, 768->d_model
        2*768 → 2*d_model → 768
        """
        super().__init__()
        self.adjust_in = nn.Sequential(
            nn.Conv2d(channels_num, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )

        self.fusion_attention = DeformableAttentionFusion_LNSpecific(
            d_model, num_encoder_layers=num_encoder_layers, num_feature_levels=num_feature_levels, DeformAttn_type="bimodal"
        )

        self.adjust_out = nn.Sequential(
            nn.Conv2d(d_model, channels_num, kernel_size=1),
            nn.GroupNorm(32, channels_num),
        )

    def forward(self, input_v, input_i):
        b, c, h, w = input_v.size()  # B, 768, 18, 18
        input_v = self.adjust_in(input_v)
        input_i = self.adjust_in(input_i)
        out = self.fusion_attention(input_v, input_i)  # B, 2*H*W, C
        out_v, out_i = torch.chunk(out, 2, 1)  # B, H*W, C
        out = out_v + out_i
        out = out.permute(0, 2, 1).view(b, -1, h, w).contiguous()
        out = self.adjust_out(out)
        return out
