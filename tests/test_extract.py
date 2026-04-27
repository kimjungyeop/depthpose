"""Regression tests for extract_bag's rotation math, subsample arithmetic,
and meta.json round-trip.

These tests do not touch pyrealsense2 — they exercise the pure-Python
arithmetic that would silently corrupt the dataset if it broke.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from depthpose.data.extract_bag import (
    file_sha256,
    rotate_image_cw,
    rotate_intrinsics,
)


# ----------------------- rotate_intrinsics -----------------------------


@pytest.fixture
def base_intr() -> dict:
    return {
        "width": 1280,
        "height": 720,
        "fx": 915.5,
        "fy": 915.5,
        "cx": 627.7,
        "cy": 365.0,
        "model": "inverse_brown_conrady",
        "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def test_rotate_intrinsics_identity(base_intr: dict) -> None:
    out = rotate_intrinsics(base_intr, 0)
    assert out == base_intr


def test_rotate_intrinsics_360_equals_zero(base_intr: dict) -> None:
    assert rotate_intrinsics(base_intr, 360) == rotate_intrinsics(base_intr, 0)


def test_rotate_intrinsics_180_swaps_principal_point(base_intr: dict) -> None:
    out = rotate_intrinsics(base_intr, 180)
    assert (out["width"], out["height"]) == (1280, 720)
    assert out["fx"] == base_intr["fx"]
    assert out["fy"] == base_intr["fy"]
    assert out["cx"] == 1280 - 1 - 627.7
    assert out["cy"] == 720 - 1 - 365.0


def test_rotate_intrinsics_90_and_270_are_inverses(base_intr: dict) -> None:
    out = rotate_intrinsics(rotate_intrinsics(base_intr, 90), 270)
    assert out == base_intr


def test_rotate_intrinsics_four_quarter_turns_is_identity(base_intr: dict) -> None:
    cur = base_intr
    for _ in range(4):
        cur = rotate_intrinsics(cur, 90)
    assert cur == base_intr


def test_rotate_intrinsics_90_swaps_dims_and_axes(base_intr: dict) -> None:
    out = rotate_intrinsics(base_intr, 90)
    assert (out["width"], out["height"]) == (720, 1280)
    assert out["fx"] == base_intr["fy"]
    assert out["fy"] == base_intr["fx"]
    # principal point: (cx, cy) -> (H-1-cy, cx)
    assert out["cx"] == pytest.approx(720 - 1 - 365.0)
    assert out["cy"] == pytest.approx(627.7)


def test_rotate_intrinsics_rejects_off_axis(base_intr: dict) -> None:
    with pytest.raises(ValueError):
        rotate_intrinsics(base_intr, 45)


# Geometric round-trip: project a known 3D point with old intrinsics,
# rotate the image, project the same 3D point with rotated intrinsics
# *with appropriately rotated camera-frame coords*, and verify the
# pixel ends up where the rotated image places the original pixel.
@pytest.mark.parametrize("rot", [90, 180, 270])
def test_rotate_intrinsics_pixel_round_trip(base_intr: dict, rot: int) -> None:
    rng = np.random.default_rng(0)
    fx, fy = base_intr["fx"], base_intr["fy"]
    cx, cy = base_intr["cx"], base_intr["cy"]
    W, H = base_intr["width"], base_intr["height"]
    for _ in range(20):
        X, Y = rng.uniform(-0.5, 0.5, size=2)
        Z = rng.uniform(0.4, 1.6)
        # Project in original frame
        u_old = fx * X / Z + cx
        v_old = fy * Y / Z + cy
        if not (0 <= u_old < W and 0 <= v_old < H):
            continue
        # Where this pixel lands in the rotated image
        if rot == 90:
            u_new_img = H - 1 - v_old
            v_new_img = u_old
        elif rot == 180:
            u_new_img = W - 1 - u_old
            v_new_img = H - 1 - v_old
        else:  # 270
            u_new_img = v_old
            v_new_img = W - 1 - u_old
        # Project the same 3D point in the rotated camera frame.
        # Convention: 90° CW image rotation ↔ 90° CW camera rotation
        # about z, so (X, Y, Z) -> (-Y, X, Z) for 90°.
        if rot == 90:
            X_new, Y_new = -Y, X
        elif rot == 180:
            X_new, Y_new = -X, -Y
        else:  # 270
            X_new, Y_new = Y, -X
        new_intr = rotate_intrinsics(base_intr, rot)
        u_pred = new_intr["fx"] * X_new / Z + new_intr["cx"]
        v_pred = new_intr["fy"] * Y_new / Z + new_intr["cy"]
        assert u_pred == pytest.approx(u_new_img, abs=1e-6)
        assert v_pred == pytest.approx(v_new_img, abs=1e-6)


# ----------------------- rotate_image_cw -------------------------------


def test_rotate_image_cw_shapes_uint8() -> None:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert rotate_image_cw(img, 0).shape == (720, 1280, 3)
    assert rotate_image_cw(img, 90).shape == (1280, 720, 3)
    assert rotate_image_cw(img, 180).shape == (720, 1280, 3)
    assert rotate_image_cw(img, 270).shape == (1280, 720, 3)


def test_rotate_image_cw_shapes_uint16_depth() -> None:
    depth = np.zeros((720, 1280), dtype=np.uint16)
    assert rotate_image_cw(depth, 0).shape == (720, 1280)
    assert rotate_image_cw(depth, 90).shape == (1280, 720)


def test_rotate_image_cw_pixel_mapping_uint8() -> None:
    """A pixel at (u=10, v=5) in W=20, H=10 should land at (u'=4, v'=10) under 90° CW."""
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[5, 10] = (1, 2, 3)
    rot = rotate_image_cw(img, 90)
    assert rot.shape == (20, 10, 3)
    # Expected new coords: (col_new, row_new) = (H-1-v, u) = (4, 10)
    assert tuple(int(c) for c in rot[10, 4]) == (1, 2, 3)


# ----------------------- file_sha256 -----------------------------------


def test_file_sha256_matches_known_hash(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    # echo -n "hello world" | sha256sum
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert file_sha256(p) == expected


# ---------------- Live-session sanity (skipped if pilot absent) --------


def _first_session_dir() -> Path | None:
    metas = sorted(Path("data/raw").glob("*/*/meta.json"))
    return metas[0].parent if metas else None


def _pilot_meta() -> dict | None:
    sd = _first_session_dir()
    if sd is None:
        return None
    return json.loads((sd / "meta.json").read_text())


@pytest.mark.skipif(_pilot_meta() is None, reason="pilot session not extracted")
def test_pilot_meta_invariants() -> None:
    m = _pilot_meta()
    assert m is not None
    assert m["saved_frames"] == len(m["frames"])
    # kept_index is a 0..N-1 contiguous range
    kept = [f["kept_index"] for f in m["frames"]]
    assert kept == list(range(m["saved_frames"]))
    # Hardware timestamps are monotonically non-decreasing
    ts = [f["hardware_timestamp_ms"] for f in m["frames"]]
    assert all(b >= a for a, b in zip(ts, ts[1:]))
    # Alignment is color→depth, so colour and depth share the same intrinsics
    assert m["alignment_target"] == "depth"
    assert m["color_intrinsics"] == m["depth_intrinsics"]
    # Depth intrinsics equal source-depth-intrinsics rotated by mount_rotation_cw_deg
    from depthpose.data.extract_bag import rotate_intrinsics
    expected = rotate_intrinsics(m["source_depth_intrinsics"], m["mount_rotation_cw_deg"])
    assert m["depth_intrinsics"] == expected
    # Depth scale is the D435 default ~1mm/unit
    assert 0.0009 < m["depth_scale_m_per_unit"] < 0.0011
    # On-disk file count matches saved_frames (rgb=png, depth=npy)
    sd = _first_session_dir()
    assert sd is not None
    rgb_dir = sd / "rgb"
    depth_dir = sd / "depth"
    assert sum(1 for p in rgb_dir.iterdir() if p.suffix == ".png") == m["saved_frames"]
    assert sum(1 for p in depth_dir.iterdir() if p.suffix == ".npy") == m["saved_frames"]


@pytest.mark.skipif(_pilot_meta() is None, reason="pilot session not extracted")
def test_pilot_first_depth_in_expected_range() -> None:
    """Pilot is a walker-mounted view of legs at 0.5–1.5 m. Median should land there."""
    sd = _first_session_dir()
    assert sd is not None
    depth = np.load(sd / "depth" / "000000.npy")
    assert depth.dtype == np.uint16
    m = _pilot_meta()
    assert m is not None
    assert depth.shape == (m["depth_intrinsics"]["height"],
                           m["depth_intrinsics"]["width"])
    nz = depth[depth > 0]
    assert nz.size > 0
    median_mm = float(np.median(nz))
    # Generous bounds: 0.1 m to 3 m
    assert 100 < median_mm < 3000
