"""2D + depth → 3D unprojection in the camera frame.

Given a 2D keypoint ``(u, v)`` in image coordinates and a depth map in
millimetres, sample the depth in a small ``kernel × kernel`` patch
around the keypoint (median of nonzero pixels — discards zero-valued
"no-return" pixels) and unproject through a pinhole model::

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z

Resulting 3D points are in the camera frame, in metres. If every pixel
in the kernel is zero, the keypoint is flagged ``depth_invalid`` and the
3D coord is left at ``[0, 0, 0]`` for the caller to filter.
"""

from __future__ import annotations

import numpy as np

Intrinsics = tuple[float, float, float, float]  # fx, fy, cx, cy


def sample_depth_median_mm(
    depth_mm: np.ndarray,
    u: float,
    v: float,
    kernel: int = 3,
    min_valid_mm: int = 1,
    max_valid_mm: int = 65534,
) -> tuple[float, bool]:
    """Median of in-range depth pixels in a kernel×kernel patch around (u, v).

    Pixels with value < ``min_valid_mm`` or > ``max_valid_mm`` are
    treated as no-return. ``min_valid_mm`` defaults to 1 so the historic
    ``> 0`` zero-skip is preserved; raise it (e.g. to 280 for D435 near-
    clip) at the call site.

    Returns ``(depth_mm_value, valid)``. ``valid=False`` if the patch
    falls outside the image or every pixel is out of range.
    """
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError(f"kernel must be odd ≥1, got {kernel}")
    if depth_mm.dtype != np.uint16:
        raise TypeError(f"depth_mm must be uint16 (raw mm), got {depth_mm.dtype}")
    if min_valid_mm < 1:
        raise ValueError(f"min_valid_mm must be ≥1 (zero is reserved), got {min_valid_mm}")
    if max_valid_mm < min_valid_mm:
        raise ValueError(f"max_valid_mm ({max_valid_mm}) must be ≥ min_valid_mm "
                         f"({min_valid_mm})")
    H, W = depth_mm.shape
    cu = int(round(u))
    cv = int(round(v))
    if not (0 <= cu < W and 0 <= cv < H):
        return 0.0, False
    half = kernel // 2
    u_lo = max(0, cu - half)
    u_hi = min(W, cu + half + 1)
    v_lo = max(0, cv - half)
    v_hi = min(H, cv + half + 1)
    patch = depth_mm[v_lo:v_hi, u_lo:u_hi]
    valid_pixels = patch[(patch >= min_valid_mm) & (patch <= max_valid_mm)]
    if valid_pixels.size == 0:
        return 0.0, False
    return float(np.median(valid_pixels)), True


def unproject(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: Intrinsics,
) -> np.ndarray:
    """Pinhole back-projection. Returns (X, Y, Z) in the same units as ``depth_m``."""
    fx, fy, cx, cy = intrinsics
    return np.array([
        (u - cx) * depth_m / fx,
        (v - cy) * depth_m / fy,
        depth_m,
    ], dtype=np.float32)


# D435 near/far range bounds (mm). Below ~280 mm the sensor is at its
# near-clip artefact range and depth values are unreliable. Above ~10 m
# the sensor is at its outdoor range limit and noise dominates.
DEFAULT_MIN_VALID_MM = 280
DEFAULT_MAX_VALID_MM = 10_000


def lift_keypoints_2d_to_3d(
    keypoints_uv: np.ndarray,           # (J, 2) float
    depth_mm: np.ndarray,               # (H, W) uint16 raw mm
    intrinsics: Intrinsics,
    median_kernel: int = 3,
    min_valid_mm: int = DEFAULT_MIN_VALID_MM,
    max_valid_mm: int = DEFAULT_MAX_VALID_MM,
    depth_scale_m_per_unit: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift J 2D keypoints to 3D using sampled depth.

    Pixels with depth outside ``[min_valid_mm, max_valid_mm]`` are
    treated as no-return — defaults bracket the D435's reliable range
    so near-clip noise (≈180 mm spurious returns at the depth-FOV edge)
    and 65535-saturation values are excluded from the median.

    Returns
    -------
    coords_3d : (J, 3) float32
        Camera-frame 3D coords in metres. Rows where ``depth_valid`` is
        ``False`` are zero-filled and should be ignored by the caller.
    depth_valid : (J,) bool
        ``True`` if depth was successfully sampled at this keypoint.
    """
    if keypoints_uv.ndim != 2 or keypoints_uv.shape[1] != 2:
        raise ValueError(f"keypoints_uv must be (J, 2), got {keypoints_uv.shape}")
    J = keypoints_uv.shape[0]
    coords_3d = np.zeros((J, 3), dtype=np.float32)
    valid = np.zeros((J,), dtype=bool)
    for j in range(J):
        u, v = float(keypoints_uv[j, 0]), float(keypoints_uv[j, 1])
        z_mm, ok = sample_depth_median_mm(
            depth_mm, u, v, kernel=median_kernel,
            min_valid_mm=min_valid_mm, max_valid_mm=max_valid_mm,
        )
        if not ok:
            continue
        z_m = z_mm * depth_scale_m_per_unit
        coords_3d[j] = unproject(u, v, z_m, intrinsics)
        valid[j] = True
    return coords_3d, valid
