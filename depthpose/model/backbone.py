"""MobileNetV2 backbone adapted for single-channel depth input.

We use ``timm`` for both the architecture and the ``in_chans=1`` first-
conv adaptation: timm's standard policy when reducing input channels is
to **sum the pretrained RGB weights along the channel axis**, which is
exactly what the brief specifies.

Returns the final stride-32 feature map (1280 channels for
``mobilenetv2_100`` at default width). The deconv head upsamples from
there to stride 4.
"""

from __future__ import annotations

from typing import Literal

import timm
import torch
from torch import nn


class DepthBackbone(nn.Module):
    """timm MobileNetV2-style backbone, 1-channel input, features-only output."""

    def __init__(
        self,
        name: Literal["mobilenetv2_100"] = "mobilenetv2_100",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.net = timm.create_model(
            name,
            pretrained=pretrained,
            in_chans=1,
            features_only=True,
            out_indices=(4,),
        )
        info = self.net.feature_info
        self.out_channels: int = info.channels()[0]
        self.out_stride: int = info.reduction()[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W). Returns the last stage (B, C, H/stride, W/stride).
        feats = self.net(x)
        return feats[0]
