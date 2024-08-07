"""
Author: Yakhyokhuja Valikhujaev
Date: 2024-08-07
Description: BiSeNet Model Implementation
Copyright (c) 2024 Yakhyokhuja Valikhujaev. All rights reserved.
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.resnet import resnet18, resnet34
from typing import Union, Optional, Tuple


class ConvBNReLU(nn.Module):
    """Standard Convolutional Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            dilation: int = 1,
            inplace: bool = True,
            bias: bool = False,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class BiSeNetOutput(nn.Module):
    """BiSeNet Output"""

    def __init__(self, in_channels: int, mid_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
        )
        self.conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_block(x)
        x = self.conv(x)
        return x


class AttentionRefinementModule(nn.Module):
    """Attention Refinement Module """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.conv_block(x)
        
        feat_shape = [int(t) for t in feat.size()[2:]]
        pool = F.avg_pool2d(feat, feat_shape)
        # pool = F.avg_pool2d(feat, feat.size()[2:]) # gives error when converting to onnx due to dynamic size

        attention = self.attention(pool)
        out = torch.mul(feat, attention)
        return out


class ContextPath(nn.Module):
    """Context Path Module or Multi-Scale Feature Pyramid Module"""

    def __init__(self, backbone_name: str = "resnet18") -> None:
        super().__init__()
        if backbone_name == "resnet18":
            self.backbone = resnet18()
        elif backbone_name == "resnet34":
            self.backbone = resnet34()
        else:
            raise Exception(f"Available backbone modules: resnet18, resnet34")

        self.arm16 = AttentionRefinementModule(in_channels=256, out_channels=128)
        self.arm32 = AttentionRefinementModule(in_channels=512, out_channels=128)
        self.conv_head32 = ConvBNReLU(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv_head16 = ConvBNReLU(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv_avg = ConvBNReLU(in_channels=512, out_channels=128, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # features from backbone
        feat8, feat16, feat32 = self.backbone(x)

        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]
        h32, w32 = feat32.size()[2:]

        feat32_shape = [int(t) for t in feat32.size()[2:]]
        avg = F.avg_pool2d(feat32, feat32_shape)
        # avg = F.avg_pool2d(feat32, feat32.size()[2:]) # gives error when converting to onnx due to dynamic size
        
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (h32, w32), mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (h16, w16), mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (h8, w8), mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    """Feature Fusion Module"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv_block = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp: Tensor, fcp: Tensor) -> Tensor:
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.conv_block(fcat)
        
        feat_shape = [int(t) for t in feat.size()[2:]]
        attention = F.avg_pool2d(feat, feat_shape)
        # attention = F.avg_pool2d(feat, feat.size()[2:]) # gives error when converting to onnx due to dynamic size
        
        attention = self.conv1(attention)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        feat_attention = torch.mul(feat, attention)
        feat_out = feat_attention + feat
        return feat_out


class BiSeNet(nn.Module):
    def __init__(self, num_classes, backbone_name="resnet18"):
        super().__init__()
        self.fpn = ContextPath(backbone_name=backbone_name)
        self.ffm = FeatureFusionModule(in_channels=256, out_channels=256)

        self.conv_out = BiSeNetOutput(in_channels=256, mid_channels=256, num_classes=num_classes)
        self.conv_out16 = BiSeNetOutput(in_channels=128, mid_channels=64, num_classes=num_classes)
        self.conv_out32 = BiSeNetOutput(in_channels=128, mid_channels=64, num_classes=num_classes)

    def forward(self, x):
        h, w = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.fpn(x)  # here return res3b1 feature
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (h, w), mode="bilinear", align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (h, w), mode="bilinear", align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (h, w), mode="bilinear", align_corners=True)

        return feat_out, feat_out16, feat_out32
