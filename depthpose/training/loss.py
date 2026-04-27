"""3D Smooth-L1 with per-joint depth-valid masking, plus optional aux 2D MSE
and optional anatomical (lateral-consistency) loss.

Per project memory: never filter whole frames, mask per-joint. The 3D
loss zero-weights rows where ``depth_valid`` is False, so a frame with
hips out of FOV still contributes a 4-joint loss from its valid knees +
ankles.

Optional auxiliary 2D heatmap MSE: target heatmap is a Gaussian peak at
the GT (u, v) in heatmap-pixel coords. Off by default
(``aux_heatmap_loss_weight=0.0``); enable if pure-3D training is
unstable. The 2D loss does NOT use the depth-valid mask — it does use
``needs_review`` (low-conf detections) so we don't fit obvious oracle
mistakes.

Optional anatomical loss: penalises any frame where the predicted hip
lateral order disagrees with the knee or ankle lateral order, which is
physically impossible for a standing/walking subject. Uses *only* the
predictions, no GT — so it acts as a self-consistency regulariser.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Joint indices into the JOINT_ORDER tuple in training_dataset.py.
# Hard-coded here to keep loss.py free of dataset imports.
_LH, _RH, _LK, _RK, _LA, _RA = 0, 1, 2, 3, 4, 5


def make_gaussian_heatmap_targets(
    uv_input: torch.Tensor,    # (B, J, 2) — input-image pixel coords
    head_stride: int,
    out_hw: tuple[int, int],   # (H_hm, W_hm)
    sigma: float = 2.0,
) -> torch.Tensor:
    """Return (B, J, H_hm, W_hm) Gaussian heatmaps centred at uv/head_stride."""
    B, J, _ = uv_input.shape
    H, W = out_hw
    device = uv_input.device
    ys = torch.arange(H, device=device, dtype=uv_input.dtype).view(1, 1, H, 1)
    xs = torch.arange(W, device=device, dtype=uv_input.dtype).view(1, 1, 1, W)
    cu = (uv_input[..., 0:1] / head_stride).unsqueeze(-1)  # (B, J, 1, 1)
    cv = (uv_input[..., 1:2] / head_stride).unsqueeze(-1)
    g = torch.exp(-((xs - cu) ** 2 + (ys - cv) ** 2) / (2.0 * sigma ** 2))
    return g  # peak value 1.0


def lateral_consistency_loss(coords_3d: torch.Tensor) -> torch.Tensor:
    """Hinge penalty when predicted L/R lateral order disagrees across joint
    pairs (hip vs knee, knee vs ankle, hip vs ankle).

    For any two joint-pairs (e.g. hips and knees), define
    ``dx_pair = x_left - x_right``. If both pairs have the same lateral
    order, ``dx_hip * dx_knee > 0``; if they disagree, ``< 0``. The
    hinge ``max(0, -dx_a * dx_b)`` is zero on consistent frames and equal
    to ``|dx_a * dx_b|`` on crossover frames. Mean over the batch and over
    the three pair combinations.

    Uses only predictions, so the loss applies on every frame regardless
    of GT availability — ideal as a soft physical-constraint regulariser.
    """
    x = coords_3d[..., 0]                        # (B, J)
    dx_hip   = x[..., _LH] - x[..., _RH]         # (B,)
    dx_knee  = x[..., _LK] - x[..., _RK]
    dx_ankle = x[..., _LA] - x[..., _RA]
    hk = F.relu(-(dx_hip * dx_knee))
    ka = F.relu(-(dx_knee * dx_ankle))
    ha = F.relu(-(dx_hip * dx_ankle))
    return ((hk + ka + ha) / 3.0).mean()


def depth_pose_loss(
    pred: dict[str, torch.Tensor],
    target_3d: torch.Tensor,
    target_uv_input: torch.Tensor,
    depth_valid: torch.Tensor,
    needs_review: torch.Tensor,
    *,
    head_stride: int,
    aux_heatmap_loss_weight: float = 0.0,
    anatomical_loss_weight: float = 0.0,
    heatmap_sigma: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Compute total loss + per-component diagnostics.

    Returns a dict with at least ``total``, ``l3d``, ``num_valid``.
    Inputs:
    - ``pred["coords_3d"]`` (B, J, 3); ``pred["heatmaps_logits"]`` (B, J, H_hm, W_hm)
    - ``target_3d`` (B, J, 3); rows masked by ``depth_valid``.
    - ``target_uv_input`` (B, J, 2); used only for the 2D aux target.
    - ``depth_valid`` (B, J) bool; ``needs_review`` (B, J) bool.
    """
    coords_3d_pred: torch.Tensor = pred["coords_3d"]
    valid = depth_valid & ~needs_review                  # (B, J)
    weight = valid.unsqueeze(-1).to(coords_3d_pred.dtype)  # (B, J, 1)

    per_elem = F.smooth_l1_loss(
        coords_3d_pred, target_3d, reduction="none", beta=0.05,
    )                                                    # (B, J, 3)
    weighted = per_elem * weight
    denom = weight.sum().clamp_min(1.0)
    l3d = weighted.sum() / denom * 3.0  # × 3 because we averaged over 3 dims

    out: dict[str, torch.Tensor] = {
        "l3d": l3d,
        "num_valid": valid.sum().detach(),
    }

    total = l3d
    if aux_heatmap_loss_weight > 0:
        hm_logits: torch.Tensor = pred["heatmaps_logits"]   # (B, J, H_hm, W_hm)
        H_hm, W_hm = hm_logits.shape[-2:]
        target_hm = make_gaussian_heatmap_targets(
            target_uv_input, head_stride=head_stride,
            out_hw=(H_hm, W_hm), sigma=heatmap_sigma,
        )
        # Per-joint mask: skip joints flagged as needs_review (oracle low-conf).
        keep_2d = (~needs_review).unsqueeze(-1).unsqueeze(-1).to(hm_logits.dtype)
        per_pix = F.mse_loss(hm_logits, target_hm, reduction="none")
        l2d = (per_pix * keep_2d).sum() / keep_2d.sum().clamp_min(1.0) / (H_hm * W_hm)
        out["l2d"] = l2d
        total = total + aux_heatmap_loss_weight * l2d

    if anatomical_loss_weight > 0:
        l_anat = lateral_consistency_loss(coords_3d_pred)
        out["l_anat"] = l_anat
        total = total + anatomical_loss_weight * l_anat

    out["total"] = total
    return out
