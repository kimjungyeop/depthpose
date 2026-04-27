"""Depth-only student: backbone + 2.5D head + per-sample intrinsics
unprojection to camera-frame 3D.

The student takes:
- ``depth``: (B, 1, H_in, W_in) tensor — single-channel depth in metres
  (zeros mark no-data pixels; we feed the raw value, the first conv +
  batch norm learns its own scaling).
- ``intrinsics_input``: (B, 4) tensor — (fx, fy, cx, cy) in **input-image
  pixel coordinates** (i.e. already scaled to match ``H_in × W_in``).

It returns a dict with:
- ``coords_3d``: (B, J, 3) — predicted (X, Y, Z) in metres, camera frame.
- ``coords_uv_input``: (B, J, 2) — predicted (u, v) in input-image pixel
  coordinates (scaled up from heatmap space by ``head_stride``).
- ``z_pred``: (B, J) — predicted Z in metres.
- ``heatmaps_logits``: (B, J, H_hm, W_hm) — pre-softmax heatmaps for the
  optional aux 2D loss.
"""

from __future__ import annotations

import torch
from torch import nn

from depthpose.model.backbone import DepthBackbone
from depthpose.model.head import DeconvHeatmapHead


class DepthPoseStudent(nn.Module):
    def __init__(
        self,
        backbone_name: str = "mobilenetv2_100",
        num_joints: int = 6,
        num_deconv: int = 3,
        deconv_channels: int = 256,
        softargmax_beta: float = 100.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = DepthBackbone(name=backbone_name, pretrained=pretrained)
        self.head = DeconvHeatmapHead(
            in_channels=self.backbone.out_channels,
            num_joints=num_joints,
            num_deconv=num_deconv,
            deconv_channels=deconv_channels,
            softargmax_beta=softargmax_beta,
        )
        # heatmap pixel = input pixel × (head_upsample / backbone_stride).
        # MobileNetV2 stride 32 + 3 deconvs = head stride 32/8 = 4.
        self.head_stride: int = self.backbone.out_stride // self.head.head_upsample
        self.num_joints = num_joints

    def forward(
        self,
        depth: torch.Tensor,
        intrinsics_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(f"depth must be (B, 1, H, W), got {tuple(depth.shape)}")
        if intrinsics_input.ndim != 2 or intrinsics_input.shape[1] != 4:
            raise ValueError(
                f"intrinsics_input must be (B, 4), got {tuple(intrinsics_input.shape)}"
            )

        feats = self.backbone(depth)
        coords_uv_hm, z_pred, hm_logits = self.head(feats)

        # heatmap-pixel → input-image pixel
        coords_uv_input = coords_uv_hm * float(self.head_stride)

        # Unproject (u, v, z) with input-scaled intrinsics
        fx = intrinsics_input[:, 0:1]   # (B, 1)
        fy = intrinsics_input[:, 1:2]
        cx = intrinsics_input[:, 2:3]
        cy = intrinsics_input[:, 3:4]
        u = coords_uv_input[..., 0]      # (B, J)
        v = coords_uv_input[..., 1]
        X = (u - cx) * z_pred / fx
        Y = (v - cy) * z_pred / fy
        coords_3d = torch.stack([X, Y, z_pred], dim=-1)  # (B, J, 3)

        return {
            "coords_3d": coords_3d,
            "coords_uv_input": coords_uv_input,
            "z_pred": z_pred,
            "heatmaps_logits": hm_logits,
        }
