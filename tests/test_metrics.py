"""Tests for eval/metrics.py — MPJPE + PCK with masking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from depthpose.eval.metrics import (
    euclidean_error_mm,
    mpjpe_overall_mm,
    mpjpe_per_joint_mm,
    pck_curve,
    pck_overall,
    pck_per_joint,
)


def test_euclidean_error_known_offset() -> None:
    """A 30 mm shift along x should produce 30 mm error per joint."""
    pred = np.zeros((2, 3, 3), dtype=np.float32)
    target = np.zeros_like(pred)
    target[..., 0] = 0.030  # 30 mm in x
    err = euclidean_error_mm(pred, target)
    assert err.shape == (2, 3)
    assert np.allclose(err, 30.0)


def test_euclidean_error_pythagorean() -> None:
    pred = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    target = np.array([[[0.030, 0.040, 0.0]]], dtype=np.float32)  # 3-4-5 triangle in mm
    err = euclidean_error_mm(pred, target)
    assert math.isclose(float(err[0, 0]), 50.0, abs_tol=1e-4)


def test_mpjpe_per_joint_masks() -> None:
    """Joint 1 has only one valid frame; joint 0 has two — confirm the masking."""
    pred = np.zeros((2, 2, 3))
    target = np.array([
        [[0.020, 0.0, 0.0], [0.040, 0.0, 0.0]],   # frame 0: errors 20mm, 40mm
        [[0.060, 0.0, 0.0], [0.080, 0.0, 0.0]],   # frame 1: errors 60mm, 80mm
    ])
    valid = np.array([[True, True], [True, False]])  # joint 1, frame 1 invalid
    out = mpjpe_per_joint_mm(pred, target, valid)
    assert out[0] == pytest.approx((20 + 60) / 2)
    assert out[1] == pytest.approx(40.0)


def test_mpjpe_per_joint_all_invalid_returns_nan() -> None:
    pred = np.zeros((2, 2, 3))
    target = np.zeros_like(pred)
    valid = np.zeros((2, 2), dtype=bool)
    out = mpjpe_per_joint_mm(pred, target, valid)
    assert np.isnan(out).all()


def test_mpjpe_overall_pools_valid_only() -> None:
    pred = np.zeros((2, 2, 3))
    target = np.zeros_like(pred)
    target[..., 0] = np.array([[0.010, 0.020], [0.030, 0.040]])
    valid = np.array([[True, True], [True, False]])
    out = mpjpe_overall_mm(pred, target, valid)
    assert out == pytest.approx((10 + 20 + 30) / 3)


def test_pck_overall_at_thresholds() -> None:
    """6 valid joints, errors 5, 10, 15, 20, 25, 30 mm.
    PCK@5 = 1/6, PCK@10 = 2/6, PCK@20 = 4/6, PCK@30 = 6/6."""
    pred = np.zeros((1, 6, 3))
    target = np.zeros_like(pred)
    target[0, :, 0] = np.array([5, 10, 15, 20, 25, 30]) / 1000.0
    valid = np.ones((1, 6), dtype=bool)
    out = pck_overall(pred, target, valid, thresholds_mm=(5, 10, 20, 30))
    assert out[5] == pytest.approx(1 / 6)
    assert out[10] == pytest.approx(2 / 6)
    assert out[20] == pytest.approx(4 / 6)
    assert out[30] == pytest.approx(6 / 6)


def test_pck_per_joint_independent() -> None:
    """Each joint scored in isolation."""
    pred = np.zeros((2, 2, 3))
    target = np.zeros_like(pred)
    target[..., 0] = np.array([[0.005, 0.015], [0.005, 0.015]])  # joint 0: 5mm always; joint 1: 15mm always
    valid = np.ones((2, 2), dtype=bool)
    out = pck_per_joint(pred, target, valid, thresholds_mm=(10, 20))
    assert out[10][0] == pytest.approx(1.0)
    assert out[10][1] == pytest.approx(0.0)
    assert out[20][0] == pytest.approx(1.0)
    assert out[20][1] == pytest.approx(1.0)


def test_pck_curve_monotonic_nondecreasing() -> None:
    rng = np.random.default_rng(0)
    pred = np.zeros((50, 6, 3))
    target = rng.normal(scale=0.01, size=(50, 6, 3))
    valid = np.ones((50, 6), dtype=bool)
    ts, pck = pck_curve(pred, target, valid, max_mm=100, step_mm=1)
    assert ts.shape == pck.shape
    assert (np.diff(pck) >= -1e-9).all()  # monotonic non-decreasing
    assert pck[0] >= 0
    assert pck[-1] <= 1.0
