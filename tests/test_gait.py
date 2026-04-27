"""Tests for gait derivations.

Synthetic signals: a known sinusoidal ankle-z and a known knee-flexion
geometry so the answers are exact.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from depthpose.eval.gait import (
    GaitMetrics,
    angle_at_vertex_deg,
    derive_gait_metrics,
)


JOINTS = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def test_angle_at_vertex_right_angle() -> None:
    a = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert angle_at_vertex_deg(a, v, b) == pytest.approx(90.0)


def test_angle_at_vertex_180() -> None:
    a = np.array([-1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])
    b = np.array([+1.0, 0.0, 0.0])
    assert angle_at_vertex_deg(a, v, b) == pytest.approx(180.0)


def test_angle_at_vertex_zero_length_is_nan() -> None:
    v = np.zeros(3)
    a = np.zeros(3)
    b = np.array([1.0, 0.0, 0.0])
    assert math.isnan(angle_at_vertex_deg(a, v, b))


def test_derive_gait_metrics_synthetic_2hz_walking() -> None:
    """Build a 5-second sequence at 30 fps with 2 Hz ankle-z oscillation
    (so cadence should be 2 steps/sec = 120 steps/min, stride period 0.5s).
    Hip/knee/ankle along a straight line so knee flexion ~180°."""
    fps = 30.0
    T = int(5 * fps)
    t = np.arange(T) / fps
    # Ankle z: 0.8 m baseline, ±0.10 m oscillation at 2 Hz
    ankle_z = 0.80 + 0.10 * np.cos(2 * np.pi * 2.0 * t)

    coords = np.zeros((T, len(JOINTS), 3), dtype=np.float32)
    # Use left-side joints
    hip_i = JOINTS.index("left_hip")
    knee_i = JOINTS.index("left_knee")
    ankle_i = JOINTS.index("left_ankle")
    # Vertical leg: hip at (0, -0.5, z_avg), knee at (0, 0, z_avg), ankle at (0, 0.5, z(t))
    # Set z to vary with time only on ankle to keep things simple
    z_avg = 0.80
    coords[:, hip_i, 1] = -0.5
    coords[:, hip_i, 2] = z_avg
    coords[:, knee_i, 1] = 0.0
    coords[:, knee_i, 2] = z_avg
    coords[:, ankle_i, 1] = 0.5
    coords[:, ankle_i, 2] = ankle_z

    valid = np.zeros((T, len(JOINTS)), dtype=bool)
    valid[:, [hip_i, knee_i, ankle_i]] = True

    g = derive_gait_metrics(coords, valid, JOINTS, fps=fps, side="left")
    assert g is not None
    # 2 Hz → 10 peaks in 5 s, give or take 1 at boundaries
    assert 8 <= g.n_steps <= 12
    # Stride period ≈ 0.5 s; allow 10% slack for boundary effects
    assert g.stride_period_s == pytest.approx(0.5, rel=0.10)
    # Cadence ~120 steps/min
    assert g.cadence_steps_per_min == pytest.approx(120.0, rel=0.10)
    # Apparent amplitude ~0.20 m (peak to trough)
    assert 0.10 <= g.apparent_step_amplitude_m <= 0.25
    # Knee close to fully extended; ankle z-oscillation introduces a small
    # bend so we expect ~178°, not exactly 180°.
    assert 175.0 <= g.knee_flexion_max_deg <= 180.0


def test_derive_gait_returns_none_when_no_peaks() -> None:
    """Constant signal → no peaks → can't derive cadence."""
    fps = 30.0
    T = int(2 * fps)
    coords = np.zeros((T, len(JOINTS), 3), dtype=np.float32)
    coords[..., 2] = 1.0  # all distances constant 1 m
    valid = np.ones((T, len(JOINTS)), dtype=bool)
    g = derive_gait_metrics(coords, valid, JOINTS, fps=fps, side="left")
    assert g is None
