"""Phase-3 gait-parameter derivations.

Given a per-frame sequence of 3D joint positions (and a per-joint
``valid`` mask), produce:

- ``cadence_steps_per_min``: from ankle-z peak detection.
- ``stride_period_s``: 1 / (steps/sec).
- ``knee_flexion_angle_deg`` time series: per-frame angle at the knee
  in the hip→knee→ankle triangle.
- ``apparent_step_amplitude_m``: peak-to-trough of ankle-z over a stride
  (rough proxy for step length in the camera frame; see caveat below).

**Caveat for stride length.** A walker-mounted camera moves with the
walker. Camera-frame ankle positions oscillate but do not translate
across the world, so true stride length cannot be recovered from camera
coords alone. The ``apparent_step_amplitude_m`` we report is the
swing-phase amplitude in the camera frame — it correlates with stride
but is not the same quantity. A real stride-length estimate would
require integrating IMU velocity from the bag's accel/gyro streams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class GaitMetrics:
    n_steps: int
    duration_s: float
    cadence_steps_per_min: float
    stride_period_s: float
    apparent_step_amplitude_m: float
    knee_flexion_min_deg: float
    knee_flexion_max_deg: float
    knee_flexion_range_deg: float


def _peak_indices(
    signal: np.ndarray,            # (T,)
    min_distance: int,
    prominence: float | None = None,
) -> np.ndarray:
    """``scipy.signal.find_peaks`` wrapper with sensible defaults."""
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got {signal.shape}")
    if prominence is None:
        # Default: 10% of the signal's full range; rejects flat plateaus.
        rng = float(signal.max() - signal.min())
        prominence = 0.10 * rng if rng > 0 else 1e9  # huge = "no peaks"
    peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence)
    return peaks


def angle_at_vertex_deg(
    a: np.ndarray, vertex: np.ndarray, b: np.ndarray,
) -> float:
    """Interior angle at ``vertex`` in the triangle a-vertex-b, in degrees."""
    va = a - vertex
    vb = b - vertex
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return float("nan")
    cos = float(np.dot(va, vb)) / (na * nb)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def derive_gait_metrics(
    coords_seq: np.ndarray,        # (T, J, 3) metres, camera frame
    valid_seq: np.ndarray,         # (T, J) bool
    joint_names: list[str],
    fps: float,
    side: str = "left",            # "left" or "right"
) -> GaitMetrics | None:
    """Compute gait metrics from one session's sequence. ``None`` if not
    enough valid frames to detect any cycle."""
    if coords_seq.shape[0] != valid_seq.shape[0]:
        raise ValueError("coords_seq and valid_seq must have same T")
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    hip_i = name_to_idx[f"{side}_hip"]
    knee_i = name_to_idx[f"{side}_knee"]
    ankle_i = name_to_idx[f"{side}_ankle"]

    T = coords_seq.shape[0]
    # Ankle Z time series (mask invalid)
    ankle_z = coords_seq[:, ankle_i, 2].astype(float)
    ankle_valid = valid_seq[:, ankle_i].astype(bool)
    # Linearly interpolate small invalid gaps so the peak picker doesn't
    # get confused; long invalid runs become NaN-padded which kills peaks.
    ankle_z_interp = ankle_z.copy()
    if (~ankle_valid).any():
        idx = np.arange(T)
        ankle_z_interp[~ankle_valid] = np.interp(
            idx[~ankle_valid], idx[ankle_valid], ankle_z[ankle_valid],
        ) if ankle_valid.any() else ankle_z[~ankle_valid]
    # Smooth with a small box filter to suppress jitter.
    k = max(3, int(round(fps / 10)) | 1)  # ~100ms window, odd
    pad = k // 2
    smoothed = np.convolve(ankle_z_interp, np.ones(k) / k, mode="same")

    min_distance = max(3, int(round(fps * 0.4)))   # min 0.4 s between peaks
    peaks = _peak_indices(smoothed, min_distance=min_distance)
    troughs = _peak_indices(-smoothed, min_distance=min_distance)
    if len(peaks) < 2:
        return None

    duration_s = (T - 1) / fps
    n_steps = len(peaks)
    stride_periods = np.diff(peaks) / fps
    stride_period_s = float(np.median(stride_periods))
    cadence_steps_per_min = 60.0 / stride_period_s if stride_period_s > 0 else 0.0
    if len(troughs) > 0:
        apparent_step_amplitude_m = float(smoothed[peaks].mean() - smoothed[troughs].mean())
    else:
        apparent_step_amplitude_m = float(smoothed[peaks].mean() - smoothed.min())

    # Knee flexion angle over time (mask frames where any of hip/knee/ankle invalid)
    flex_valid = (
        valid_seq[:, hip_i] & valid_seq[:, knee_i] & valid_seq[:, ankle_i]
    )
    flex = np.full(T, np.nan, dtype=float)
    for t in range(T):
        if flex_valid[t]:
            flex[t] = angle_at_vertex_deg(
                coords_seq[t, hip_i], coords_seq[t, knee_i], coords_seq[t, ankle_i],
            )
    flex_clean = flex[~np.isnan(flex)]
    if flex_clean.size == 0:
        knee_min = knee_max = knee_range = float("nan")
    else:
        knee_min = float(flex_clean.min())
        knee_max = float(flex_clean.max())
        knee_range = knee_max - knee_min

    return GaitMetrics(
        n_steps=int(n_steps),
        duration_s=float(duration_s),
        cadence_steps_per_min=float(cadence_steps_per_min),
        stride_period_s=stride_period_s,
        apparent_step_amplitude_m=apparent_step_amplitude_m,
        knee_flexion_min_deg=knee_min,
        knee_flexion_max_deg=knee_max,
        knee_flexion_range_deg=knee_range,
    )
