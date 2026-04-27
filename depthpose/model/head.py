"""2.5D heatmap + depth-offset head with soft-argmax.

Architecture (SimpleBaselines / 3D-pose-baseline style):

- 3 deconv blocks upsample stride-32 features back to stride 4.
- A final 1×1 conv produces 2J channels:
  * J heatmap channels: per-joint spatial likelihood.
  * J depth-offset channels: per-pixel z prediction in metres.
- Soft-argmax on the heatmaps gives subpixel ``(u_h, v_h)`` in heatmap
  pixel coordinates. The depth-offset is then aggregated under the
  same softmax weights to give a single scalar ``z`` per joint.

The forward returns:
- ``coords_uv_heatmap``: (B, J, 2) — soft-argmax expected (u, v) in
  heatmap-pixel coordinates.
- ``z_pred``: (B, J) — heatmap-weighted expected ``z`` in metres.
- ``heatmaps_logits``: (B, J, H_hm, W_hm) — pre-softmax for aux loss.

Unprojection to camera-frame 3D is in ``student.py`` because it needs
the per-sample intrinsics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DeconvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))


class DeconvHeatmapHead(nn.Module):
    """Deconv-3 head producing (heatmap, z_offset) per joint."""

    def __init__(
        self,
        in_channels: int,
        num_joints: int,
        num_deconv: int = 3,
        deconv_channels: int = 256,
        softargmax_beta: float = 100.0,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.softargmax_beta = softargmax_beta
        layers: list[nn.Module] = []
        c = in_channels
        for _ in range(num_deconv):
            layers.append(DeconvBlock(c, deconv_channels))
            c = deconv_channels
        self.upsampler = nn.Sequential(*layers)
        # 2J channels: J heatmaps + J z-offsets.
        self.final = nn.Conv2d(c, num_joints * 2, kernel_size=1)
        # The deconv stride relative to the backbone output:
        self.head_upsample = 2 ** num_deconv  # e.g. 8

    def forward(
        self,
        feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.upsampler(feats)
        x = self.final(x)             # (B, 2J, H_hm, W_hm)
        B, _, H, W = x.shape
        J = self.num_joints
        heatmaps_logits = x[:, :J]    # (B, J, H, W)
        z_offsets = x[:, J:]          # (B, J, H, W) — pre-aggregation z map
        # Soft-argmax: softmax over flattened spatial dims, then expected (u, v)
        flat = (heatmaps_logits * self.softargmax_beta).flatten(2)  # (B, J, HW)
        weights = F.softmax(flat, dim=-1).reshape(B, J, H, W)
        device = heatmaps_logits.device
        ys = torch.arange(H, device=device, dtype=heatmaps_logits.dtype)
        xs = torch.arange(W, device=device, dtype=heatmaps_logits.dtype)
        # Expected (u, v) in heatmap-pixel coords
        exp_u = (weights.sum(dim=2) * xs).sum(dim=-1)   # (B, J)
        exp_v = (weights.sum(dim=3) * ys).sum(dim=-1)   # (B, J)
        coords_uv = torch.stack([exp_u, exp_v], dim=-1)  # (B, J, 2)
        # Expected z under the same weights
        z_pred = (weights * z_offsets).sum(dim=(2, 3))  # (B, J)
        return coords_uv, z_pred, heatmaps_logits
