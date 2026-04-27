"""3D pose evaluation: MPJPE + per-joint MPJPE + PCK.

Operates on numpy arrays so the same code can score either the trained
student's predictions or a synthetic baseline. Per project memory:
**mask per-joint with depth_valid; never filter whole frames**. Joints
where the target is invalid are simply excluded from that joint's mean.
"""

from __future__ import annotations

import numpy as np


def euclidean_error_mm(
    pred_xyz_m: np.ndarray,         # (N, J, 3) metres
    target_xyz_m: np.ndarray,       # (N, J, 3) metres
) -> np.ndarray:
    """Per-frame, per-joint L2 error in millimetres. Shape (N, J)."""
    if pred_xyz_m.shape != target_xyz_m.shape:
        raise ValueError(
            f"shape mismatch: pred {pred_xyz_m.shape} vs target {target_xyz_m.shape}"
        )
    diff = pred_xyz_m - target_xyz_m
    return np.linalg.norm(diff, axis=-1) * 1000.0


def mpjpe_per_joint_mm(
    pred_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    valid: np.ndarray,              # (N, J) bool
) -> np.ndarray:
    """Mean L2 error per joint in mm; masked-mean over valid rows. Shape (J,)."""
    err = euclidean_error_mm(pred_xyz_m, target_xyz_m)        # (N, J)
    valid = valid.astype(bool)
    counts = valid.sum(axis=0).clip(min=1)
    sums = (err * valid).sum(axis=0)
    out = sums / counts
    out[valid.sum(axis=0) == 0] = np.nan
    return out


def mpjpe_overall_mm(
    pred_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    valid: np.ndarray,
) -> float:
    err = euclidean_error_mm(pred_xyz_m, target_xyz_m)
    valid = valid.astype(bool)
    if not valid.any():
        return float("nan")
    return float(err[valid].mean())


def pck_per_joint(
    pred_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    valid: np.ndarray,
    thresholds_mm: tuple[int, ...] = (5, 10, 20, 50),
) -> dict[int, np.ndarray]:
    """Per-joint PCK at each threshold. ``out[t]`` is shape ``(J,)`` ∈ [0, 1]."""
    err = euclidean_error_mm(pred_xyz_m, target_xyz_m)
    valid = valid.astype(bool)
    out: dict[int, np.ndarray] = {}
    for t in thresholds_mm:
        hits = (err <= t) & valid
        counts = valid.sum(axis=0).clip(min=1)
        out[t] = hits.sum(axis=0) / counts
    return out


def pck_overall(
    pred_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    valid: np.ndarray,
    thresholds_mm: tuple[int, ...] = (5, 10, 20, 50),
) -> dict[int, float]:
    """Single PCK value pooled across all valid joint predictions."""
    err = euclidean_error_mm(pred_xyz_m, target_xyz_m)
    valid = valid.astype(bool)
    n_valid = int(valid.sum())
    out: dict[int, float] = {}
    for t in thresholds_mm:
        hits = ((err <= t) & valid).sum()
        out[t] = float(hits) / max(n_valid, 1)
    return out


def pck_curve(
    pred_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    valid: np.ndarray,
    max_mm: int = 100,
    step_mm: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """``(thresholds_mm, pck)`` arrays for plotting a PCK curve."""
    err = euclidean_error_mm(pred_xyz_m, target_xyz_m)
    valid = valid.astype(bool)
    if not valid.any():
        ts = np.arange(1, max_mm + 1, step_mm)
        return ts, np.full_like(ts, np.nan, dtype=float)
    err_valid = err[valid]
    thresholds = np.arange(0, max_mm + 1, step_mm, dtype=int)
    pck = np.array([float((err_valid <= t).mean()) for t in thresholds])
    return thresholds, pck
