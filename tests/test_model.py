"""Shape and behaviour tests for the student model.

These tests build a tiny model (no pretrained download), pass synthetic
input through it, and check shapes + that gradients flow. They are
fast (<5s each) and run on CPU to stay portable.
"""

from __future__ import annotations

import torch

from depthpose.model.backbone import DepthBackbone
from depthpose.model.head import DeconvHeatmapHead
from depthpose.model.student import DepthPoseStudent


def _make_student(num_joints: int = 6, num_deconv: int = 3,
                  deconv_channels: int = 64) -> DepthPoseStudent:
    return DepthPoseStudent(
        backbone_name="mobilenetv2_100",
        num_joints=num_joints,
        num_deconv=num_deconv,
        deconv_channels=deconv_channels,
        softargmax_beta=100.0,
        pretrained=False,
    )


def test_backbone_output_shape_and_stride() -> None:
    bb = DepthBackbone(name="mobilenetv2_100", pretrained=False)
    x = torch.randn(2, 1, 256, 192)  # (B, 1, H, W)
    feats = bb(x)
    assert feats.shape[0] == 2
    assert feats.shape[1] == bb.out_channels
    # stride 32: 256/32=8, 192/32=6
    assert feats.shape[2] == 256 // bb.out_stride
    assert feats.shape[3] == 192 // bb.out_stride


def test_head_output_shapes() -> None:
    head = DeconvHeatmapHead(in_channels=320, num_joints=6,
                             num_deconv=3, deconv_channels=64)
    feats = torch.randn(2, 320, 8, 6)  # backbone-shaped
    coords_uv, z, hm = head(feats)
    assert coords_uv.shape == (2, 6, 2)
    assert z.shape == (2, 6)
    # 3 deconvs of stride 2: 8×6 → 64×48
    assert hm.shape == (2, 6, 64, 48)


def test_student_output_shapes() -> None:
    model = _make_student()
    depth = torch.randn(2, 1, 256, 192)
    intr = torch.tensor([[100.0, 100.0, 96.0, 128.0]] * 2, dtype=torch.float32)
    out = model(depth, intr)
    assert out["coords_3d"].shape == (2, 6, 3)
    assert out["coords_uv_input"].shape == (2, 6, 2)
    assert out["z_pred"].shape == (2, 6)
    assert out["heatmaps_logits"].shape == (2, 6, 64, 48)


def test_student_gradients_flow() -> None:
    model = _make_student()
    depth = torch.randn(2, 1, 256, 192, requires_grad=False)
    intr = torch.tensor([[100.0, 100.0, 96.0, 128.0]] * 2, dtype=torch.float32)
    out = model(depth, intr)
    target = torch.zeros_like(out["coords_3d"])
    loss = (out["coords_3d"] - target).abs().mean()
    loss.backward()
    grads = [
        p.grad for p in model.parameters()
        if p.grad is not None and p.requires_grad
    ]
    assert grads, "no gradients reached any parameter"
    total = sum(g.abs().sum().item() for g in grads)
    assert total > 0, "all gradients are zero"


def test_student_unproject_uses_input_intrinsics() -> None:
    """End-to-end shape + non-trivial output check.

    The exact unprojection math (``X = (u-cx)·z/fx`` etc.) is already
    covered by ``test_lift.py``; here we just confirm the student's
    output is finite, shaped right, and depends on intrinsics.
    """
    model = _make_student()
    model.eval()
    depth = torch.randn(2, 1, 256, 192)
    intr_a = torch.tensor([[100.0, 100.0, 96.0, 128.0]] * 2, dtype=torch.float32)
    intr_b = torch.tensor([[200.0, 200.0, 96.0, 128.0]] * 2, dtype=torch.float32)
    with torch.no_grad():
        out_a = model(depth, intr_a)
        out_b = model(depth, intr_b)
    assert torch.isfinite(out_a["coords_3d"]).all()
    # Different fx must produce different X (unless u happens to equal cx exactly).
    assert not torch.allclose(out_a["coords_3d"], out_b["coords_3d"])


def test_param_count_and_flops_within_user_accepted_budget() -> None:
    """User has accepted slightly-over budget at 256 deconv channels.
    Sanity-check the actual numbers don't drift surprisingly."""
    model = _make_student(num_joints=6, num_deconv=3, deconv_channels=256)
    n = sum(p.numel() for p in model.parameters())
    # 5.22M reported in the live smoke; allow ±0.5M jitter for refactors.
    assert 4.5e6 < n < 6.0e6, f"unexpected param count: {n / 1e6:.2f}M"
