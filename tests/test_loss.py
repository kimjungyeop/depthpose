"""Tests for depth_pose_loss + heatmap target builder."""

from __future__ import annotations

import torch

from depthpose.training.loss import depth_pose_loss, make_gaussian_heatmap_targets


def _make_pred(B: int, J: int, H_hm: int, W_hm: int) -> dict[str, torch.Tensor]:
    return {
        "coords_3d": torch.zeros(B, J, 3, requires_grad=True),
        "heatmaps_logits": torch.zeros(B, J, H_hm, W_hm, requires_grad=True),
    }


def test_l3d_zero_when_pred_equals_target() -> None:
    pred = _make_pred(2, 6, 64, 48)
    target = torch.zeros(2, 6, 3)
    valid = torch.ones(2, 6, dtype=torch.bool)
    review = torch.zeros(2, 6, dtype=torch.bool)
    out = depth_pose_loss(pred, target, torch.zeros(2, 6, 2), valid, review,
                          head_stride=4)
    assert out["l3d"].item() == 0.0


def test_l3d_masks_invalid_joints() -> None:
    """A frame with all hips invalid: only 4 joints contribute."""
    pred = _make_pred(2, 6, 64, 48)
    pred["coords_3d"] = torch.ones(2, 6, 3, requires_grad=True) * 0.1  # 10cm error per dim
    target = torch.zeros(2, 6, 3)
    valid = torch.tensor([[False, False, True, True, True, True]] * 2)
    review = torch.zeros(2, 6, dtype=torch.bool)
    out = depth_pose_loss(pred, target, torch.zeros(2, 6, 2), valid, review,
                          head_stride=4)
    # Smooth-L1(β=0.05) at err=0.1 → 0.075 per element.
    # 24 valid elements (2 batches × 4 joints × 3 dims), weight.sum()=8 (B×J),
    # l3d = (24·0.075)/8 · 3 = 0.675.
    assert 0.6 < out["l3d"].item() < 0.7
    assert int(out["num_valid"]) == 4 * 2  # 4 joints × 2 batch


def test_l3d_zero_when_no_valid_joints() -> None:
    pred = _make_pred(1, 6, 64, 48)
    target = torch.zeros(1, 6, 3)
    valid = torch.zeros(1, 6, dtype=torch.bool)
    review = torch.zeros(1, 6, dtype=torch.bool)
    out = depth_pose_loss(pred, target, torch.zeros(1, 6, 2), valid, review,
                          head_stride=4)
    assert out["l3d"].item() == 0.0
    assert int(out["num_valid"]) == 0


def test_needs_review_excludes_joint() -> None:
    """A joint with needs_review=True should not contribute even if depth_valid."""
    pred = _make_pred(1, 6, 64, 48)
    pred["coords_3d"] = torch.ones(1, 6, 3, requires_grad=True) * 1.0
    target = torch.zeros(1, 6, 3)
    valid = torch.ones(1, 6, dtype=torch.bool)
    # First joint flagged needs_review
    review = torch.tensor([[True, False, False, False, False, False]])
    out = depth_pose_loss(pred, target, torch.zeros(1, 6, 2), valid, review,
                          head_stride=4)
    assert int(out["num_valid"]) == 5


def test_aux_2d_loss_only_when_weight_positive() -> None:
    pred = _make_pred(1, 6, 64, 48)
    target = torch.zeros(1, 6, 3)
    target_uv = torch.full((1, 6, 2), 100.0)
    valid = torch.ones(1, 6, dtype=torch.bool)
    review = torch.zeros(1, 6, dtype=torch.bool)
    out0 = depth_pose_loss(pred, target, target_uv, valid, review,
                           head_stride=4, aux_heatmap_loss_weight=0.0)
    assert "l2d" not in out0

    out1 = depth_pose_loss(pred, target, target_uv, valid, review,
                           head_stride=4, aux_heatmap_loss_weight=0.1)
    assert "l2d" in out1
    assert out1["total"].item() > out1["l3d"].item()


def test_make_gaussian_heatmap_targets_peak_at_uv() -> None:
    """Gaussian peak (value 1.0) should land at (u/stride, v/stride)."""
    target_uv = torch.tensor([[[16.0, 8.0]]])  # B=1, J=1, (u, v)
    hm = make_gaussian_heatmap_targets(target_uv, head_stride=4,
                                       out_hw=(8, 8), sigma=1.0)
    assert hm.shape == (1, 1, 8, 8)
    # peak at (u/4, v/4) = (4, 2) → (col=4, row=2)
    peak_idx = hm.flatten().argmax()
    row, col = divmod(int(peak_idx), 8)
    assert (row, col) == (2, 4)
    assert hm[0, 0, 2, 4] == 1.0


def test_loss_total_is_finite_and_backward_succeeds() -> None:
    pred = _make_pred(2, 6, 64, 48)
    target = torch.randn(2, 6, 3) * 0.5
    target_uv = torch.rand(2, 6, 2) * 100
    valid = torch.ones(2, 6, dtype=torch.bool)
    review = torch.zeros(2, 6, dtype=torch.bool)
    out = depth_pose_loss(pred, target, target_uv, valid, review,
                          head_stride=4, aux_heatmap_loss_weight=0.1)
    assert torch.isfinite(out["total"])
    out["total"].backward()
    assert pred["coords_3d"].grad is not None
    assert pred["heatmaps_logits"].grad is not None
