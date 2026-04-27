"""Tests for depth sampling and pinhole unprojection."""

from __future__ import annotations

import numpy as np
import pytest

from depthpose.oracle.lift import (
    lift_keypoints_2d_to_3d,
    sample_depth_median_mm,
    unproject,
)


# ---------------------------- unproject -------------------------------


def test_unproject_principal_point_at_z_lies_on_optical_axis() -> None:
    intr = (100.0, 100.0, 50.0, 60.0)
    p = unproject(50.0, 60.0, 1.5, intr)
    np.testing.assert_allclose(p, [0.0, 0.0, 1.5])


def test_unproject_known_offset() -> None:
    intr = (200.0, 200.0, 100.0, 100.0)  # fx=fy=200, cx=cy=100
    # pixel offset of (+10, +20) at z=1.0 → world offset (0.05, 0.10, 1.0)
    p = unproject(110.0, 120.0, 1.0, intr)
    np.testing.assert_allclose(p, [0.05, 0.10, 1.0], atol=1e-6)


def test_unproject_round_trip_random_points() -> None:
    """Project a random 3D point with intrinsics, then unproject — should round-trip."""
    rng = np.random.default_rng(42)
    intr = (300.0, 305.0, 320.0, 240.0)
    fx, fy, cx, cy = intr
    for _ in range(200):
        X = rng.uniform(-1.0, 1.0)
        Y = rng.uniform(-1.0, 1.0)
        Z = rng.uniform(0.4, 2.5)
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        p = unproject(u, v, Z, intr)
        np.testing.assert_allclose(p, [X, Y, Z], atol=1e-5)


# -------------------------- sample_depth -----------------------------


def _depth(arr: list[list[int]]) -> np.ndarray:
    return np.array(arr, dtype=np.uint16)


def test_sample_depth_median_simple_3x3() -> None:
    d = _depth([
        [0, 0, 0, 0, 0],
        [0, 100, 200, 300, 0],
        [0, 400, 500, 600, 0],
        [0, 700, 800, 900, 0],
        [0, 0, 0, 0, 0],
    ])
    # Centred on (u=2, v=2): patch is [[100,200,300],[400,500,600],[700,800,900]]
    # median is 500.
    z, ok = sample_depth_median_mm(d, 2, 2, kernel=3)
    assert ok and z == 500.0


def test_sample_depth_median_skips_zeros() -> None:
    d = _depth([
        [0, 0, 0],
        [0, 100, 0],
        [0, 0, 0],
    ])
    # Only the center pixel is nonzero; median of {100} = 100.
    z, ok = sample_depth_median_mm(d, 1, 1, kernel=3)
    assert ok and z == 100.0


def test_sample_depth_median_all_zero_invalid() -> None:
    d = np.zeros((5, 5), dtype=np.uint16)
    z, ok = sample_depth_median_mm(d, 2, 2, kernel=3)
    assert not ok and z == 0.0


def test_sample_depth_median_out_of_bounds_invalid() -> None:
    d = _depth([[100, 200], [300, 400]])
    z, ok = sample_depth_median_mm(d, -1, 0, kernel=3)
    assert not ok and z == 0.0
    z, ok = sample_depth_median_mm(d, 5, 5, kernel=3)
    assert not ok and z == 0.0


def test_sample_depth_median_clips_at_image_border() -> None:
    d = _depth([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
    ])
    # Centred on top-left corner (0, 0): patch clips to [[10, 20], [40, 50]]
    # → median of {10, 20, 40, 50} = 30.
    z, ok = sample_depth_median_mm(d, 0, 0, kernel=3)
    assert ok and z == 30.0


def test_sample_depth_median_rejects_even_kernel() -> None:
    d = np.ones((3, 3), dtype=np.uint16)
    with pytest.raises(ValueError):
        sample_depth_median_mm(d, 1, 1, kernel=2)


def test_sample_depth_median_rejects_wrong_dtype() -> None:
    d = np.ones((3, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        sample_depth_median_mm(d, 1, 1, kernel=3)


def test_sample_depth_median_min_valid_filters_near_clip() -> None:
    """Near-clip pixels (e.g. 180 mm sensor noise) should drop out at min_valid=280."""
    d = _depth([
        [180, 180, 180],
        [180, 800, 180],
        [180, 180, 180],
    ])
    z, ok = sample_depth_median_mm(d, 1, 1, kernel=3, min_valid_mm=280)
    assert ok and z == 800.0


def test_sample_depth_median_min_valid_all_below_invalidates() -> None:
    d = _depth([[180, 200, 220], [240, 260, 270], [180, 200, 220]])
    z, ok = sample_depth_median_mm(d, 1, 1, kernel=3, min_valid_mm=280)
    assert not ok and z == 0.0


def test_sample_depth_median_max_valid_filters_saturation() -> None:
    """65535 (uint16 max) is sensor saturation; must drop out at the cap."""
    d = _depth([
        [65535, 65535, 65535],
        [65535,   900, 65535],
        [65535, 65535, 65535],
    ])
    z, ok = sample_depth_median_mm(d, 1, 1, kernel=3, min_valid_mm=280, max_valid_mm=10_000)
    assert ok and z == 900.0


def test_sample_depth_median_rejects_min_valid_zero() -> None:
    d = _depth([[100, 200], [300, 400]])
    with pytest.raises(ValueError):
        sample_depth_median_mm(d, 0, 0, kernel=1, min_valid_mm=0)


# ------------------------ lift_keypoints -----------------------------


def test_lift_keypoints_round_trip_with_synthetic_depth() -> None:
    """Place a 3D point at known camera-frame coords; build a depth frame
    where the projected pixel is set to that depth; lift back; verify
    unprojection recovers the original 3D coordinates."""
    intr = (300.0, 300.0, 320.0, 240.0)
    fx, fy, cx, cy = intr
    H, W = 480, 640

    rng = np.random.default_rng(0)
    expected = []
    keypoints = []
    depth = np.zeros((H, W), dtype=np.uint16)

    for _ in range(10):
        X = rng.uniform(-0.3, 0.3)
        Y = rng.uniform(-0.3, 0.3)
        Z = rng.uniform(0.5, 1.5)
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        ui, vi = int(round(u)), int(round(v))
        # set a 3x3 region around the projected pixel to Z*1000 (mm)
        z_mm = int(round(Z * 1000))
        depth[max(0, vi - 1):vi + 2, max(0, ui - 1):ui + 2] = z_mm
        keypoints.append([u, v])
        expected.append([X, Y, Z])

    coords_3d, valid = lift_keypoints_2d_to_3d(
        np.array(keypoints, dtype=np.float32),
        depth,
        intr,
    )
    assert valid.all()
    # Tolerance ~1 mm because we rounded depth to integer mm.
    np.testing.assert_allclose(coords_3d, np.array(expected), atol=2e-3)


def test_lift_keypoints_invalid_pixel_zero_filled() -> None:
    intr = (300.0, 300.0, 100.0, 100.0)
    depth = np.zeros((200, 200), dtype=np.uint16)
    depth[50, 50] = 1000  # only one nonzero pixel
    keypoints = np.array([[50.0, 50.0], [10.0, 10.0]], dtype=np.float32)
    coords_3d, valid = lift_keypoints_2d_to_3d(keypoints, depth, intr, median_kernel=1)
    assert valid.tolist() == [True, False]
    np.testing.assert_allclose(coords_3d[1], [0.0, 0.0, 0.0])


def test_lift_keypoints_default_min_valid_280_drops_near_clip() -> None:
    """Default bounds should drop a 180-mm near-clip noise reading and keep a 1000-mm one."""
    intr = (300.0, 300.0, 100.0, 100.0)
    depth = np.zeros((200, 200), dtype=np.uint16)
    depth[50, 50] = 180   # near-clip noise
    depth[80, 80] = 1000  # real reading
    keypoints = np.array([[50.0, 50.0], [80.0, 80.0]], dtype=np.float32)
    coords_3d, valid = lift_keypoints_2d_to_3d(keypoints, depth, intr, median_kernel=1)
    assert valid.tolist() == [False, True]
    assert coords_3d[1, 2] == pytest.approx(1.0)


def test_lift_keypoints_rejects_bad_input_shape() -> None:
    intr = (1.0, 1.0, 0.0, 0.0)
    depth = np.zeros((10, 10), dtype=np.uint16)
    with pytest.raises(ValueError):
        lift_keypoints_2d_to_3d(np.array([1, 2, 3], dtype=np.float32), depth, intr)
