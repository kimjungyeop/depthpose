"""Tests for skeleton-swap detection."""

from __future__ import annotations

import pandas as pd
import pytest

from depthpose.oracle.quality import (
    consistent_frame_indices,
    is_frame_skeleton_consistent,
)


def _frame(rows: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Build a parquet-shaped frame with given (u, v) per joint name."""
    return pd.DataFrame([
        {"frame_index": 0, "joint_name": n, "u_px": u, "v_px": v}
        for n, (u, v) in rows.items()
    ])


def test_consistent_skeleton_left_lower_u() -> None:
    rows = _frame({
        "left_hip":   (100.0, 200.0),
        "right_hip":  (300.0, 200.0),
        "left_knee":  (110.0, 350.0),
        "right_knee": (290.0, 350.0),
        "left_ankle": (120.0, 500.0),
        "right_ankle":(280.0, 500.0),
    })
    assert is_frame_skeleton_consistent(rows) is True


def test_consistent_skeleton_left_higher_u() -> None:
    """Same body, just mirrored. Still consistent."""
    rows = _frame({
        "left_hip":   (300.0, 200.0),
        "right_hip":  (100.0, 200.0),
        "left_knee":  (290.0, 350.0),
        "right_knee": (110.0, 350.0),
        "left_ankle": (280.0, 500.0),
        "right_ankle":(120.0, 500.0),
    })
    assert is_frame_skeleton_consistent(rows) is True


def test_swap_at_knee_detected() -> None:
    """Hips and ankles agree but knees swap → cross between hip and knee."""
    rows = _frame({
        "left_hip":   (100.0, 200.0),
        "right_hip":  (300.0, 200.0),
        "left_knee":  (290.0, 350.0),  # swap: left_knee ended up on the right
        "right_knee": (110.0, 350.0),
        "left_ankle": (120.0, 500.0),
        "right_ankle":(280.0, 500.0),
    })
    assert is_frame_skeleton_consistent(rows) is False


def test_swap_at_ankle_detected() -> None:
    rows = _frame({
        "left_hip":   (100.0, 200.0),
        "right_hip":  (300.0, 200.0),
        "left_knee":  (110.0, 350.0),
        "right_knee": (290.0, 350.0),
        "left_ankle": (280.0, 500.0),  # swap at ankle
        "right_ankle":(120.0, 500.0),
    })
    assert is_frame_skeleton_consistent(rows) is False


def test_only_hips_present_returns_consistent() -> None:
    rows = _frame({
        "left_hip":  (100.0, 200.0),
        "right_hip": (300.0, 200.0),
    })
    assert is_frame_skeleton_consistent(rows) is True


def test_consistent_frame_indices_aggregates_per_frame() -> None:
    df = pd.DataFrame([
        # frame 0: consistent
        {"frame_index": 0, "joint_name": "left_hip",   "u_px": 100, "v_px": 200},
        {"frame_index": 0, "joint_name": "right_hip",  "u_px": 300, "v_px": 200},
        {"frame_index": 0, "joint_name": "left_knee",  "u_px": 110, "v_px": 350},
        {"frame_index": 0, "joint_name": "right_knee", "u_px": 290, "v_px": 350},
        {"frame_index": 0, "joint_name": "left_ankle", "u_px": 120, "v_px": 500},
        {"frame_index": 0, "joint_name": "right_ankle","u_px": 280, "v_px": 500},
        # frame 1: swap at knee
        {"frame_index": 1, "joint_name": "left_hip",   "u_px": 100, "v_px": 200},
        {"frame_index": 1, "joint_name": "right_hip",  "u_px": 300, "v_px": 200},
        {"frame_index": 1, "joint_name": "left_knee",  "u_px": 290, "v_px": 350},
        {"frame_index": 1, "joint_name": "right_knee", "u_px": 110, "v_px": 350},
        {"frame_index": 1, "joint_name": "left_ankle", "u_px": 120, "v_px": 500},
        {"frame_index": 1, "joint_name": "right_ankle","u_px": 280, "v_px": 500},
    ])
    out = consistent_frame_indices(df)
    assert out == {0: True, 1: False}
