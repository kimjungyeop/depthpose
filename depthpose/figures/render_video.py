"""Render a side-by-side teacher/student tracking video for one session,
with a live cadence and stride-period HUD on each panel.

Layout (per frame):
  ┌─────────────────────────┬─────────────────────────┐
  │ RGB + ViTPose++ skeleton│ Depth colormap +        │
  │  (oracle / "teacher")   │  student 3D-projected   │
  │                         │  skeleton (cyan)        │
  │  [cadence / stride HUD] │  [cadence / stride HUD] │
  └─────────────────────────┴─────────────────────────┘
                with a per-frame caption strip on top.

The HUD on each panel shows the running cadence and stride period
computed *causally* from the ankle-z series seen so far on that pipeline
(oracle: parquet z_m; student: model output z), using the same
peak-detection logic as ``depthpose.eval.gait``.

Output: ``reports/videos/<subject>_<session>.mp4`` at 30 fps, mp4v codec.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import typer
from scipy.signal import find_peaks

from depthpose.data.dataset import WalkerSession
from depthpose.data.training_dataset import JOINT_ORDER, TrainingSession
from depthpose.model.student import DepthPoseStudent
from depthpose.training.config import Config

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


# ---- drawing constants ------------------------------------------------------

_DEPTH_VIS_LO_MM = 300
_DEPTH_VIS_HI_MM = 1800

# Oracle (teacher): warm reds
_ORACLE_LEFT  = (50, 255, 100)   # green
_ORACLE_RIGHT = (50, 220, 255)   # yellow-green
_ORACLE_LINE  = (240, 240, 240)

# Student: cool cyans
_STUDENT_LEFT  = (255, 230, 80)   # cyan
_STUDENT_RIGHT = (200, 100, 255)  # magenta
_STUDENT_LINE  = (200, 230, 255)

_SKELETON: list[tuple[str, str]] = [
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_hip", "right_hip"),
]


def _depth_colormap(depth_mm: np.ndarray) -> np.ndarray:
    norm = np.clip(depth_mm, _DEPTH_VIS_LO_MM, _DEPTH_VIS_HI_MM).astype(np.float32)
    norm = ((norm - _DEPTH_VIS_LO_MM) / (_DEPTH_VIS_HI_MM - _DEPTH_VIS_LO_MM) * 255).astype(np.uint8)
    out = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    out[depth_mm == 0] = (40, 40, 40)
    return out


def _draw_skeleton(
    img: np.ndarray,
    name_to_uv: dict[str, tuple[int, int]],
    valid_mask: dict[str, bool],
    *, c_left: tuple[int, int, int], c_right: tuple[int, int, int],
    c_line: tuple[int, int, int],
) -> np.ndarray:
    out = img.copy()
    for a, b in _SKELETON:
        if a in name_to_uv and b in name_to_uv and valid_mask.get(a, True) and valid_mask.get(b, True):
            cv2.line(out, name_to_uv[a], name_to_uv[b], c_line, 2, cv2.LINE_AA)
    for name, (u, v) in name_to_uv.items():
        if not valid_mask.get(name, True):
            # invalid joint: small grey ring only
            cv2.circle(out, (u, v), 6, (90, 90, 90), 1, cv2.LINE_AA)
            continue
        c = c_left if name.startswith("left") else c_right
        cv2.circle(out, (u, v), 7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (u, v), 5, c, -1, cv2.LINE_AA)
    return out


def _project_to_depth(
    coords_3d: np.ndarray,           # (J, 3) metres, camera frame
    intr: dict,                      # rotation-baked depth intrinsics
) -> tuple[dict[str, tuple[int, int]], dict[str, bool]]:
    """Project model XYZ → (u, v) in saved depth-image pixel space."""
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    W, H = intr["width"], intr["height"]
    out: dict[str, tuple[int, int]] = {}
    valid: dict[str, bool] = {}
    for i, name in enumerate(JOINT_ORDER):
        x, y, z = coords_3d[i]
        if z < 0.1 or z > 5.0:
            continue
        u = int(round(fx * x / z + cx))
        v = int(round(fy * y / z + cy))
        out[name] = (u, v)
        valid[name] = (0 <= u < W) and (0 <= v < H)
    return out, valid


def _strip_caption(width: int, height: int, text: str) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, text, (8, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _running_cadence_stride(
    ankle_z: np.ndarray,           # (T,) float
    valid: np.ndarray,             # (T,) bool
    fps: float,
) -> tuple[float, float]:
    """Causal cadence + stride period from ankle-z time series so far.

    Mirrors the peak-detection in ``depthpose.eval.gait.derive_gait_metrics``:
    interpolate small invalid gaps, smooth with a ~100 ms box, then
    ``scipy.signal.find_peaks`` with min-distance 0.4 s and
    prominence ≥ 10 % of the signal range.

    Returns (cadence_steps_per_min, stride_period_s); each is NaN if
    fewer than 2 peaks have been observed yet.
    """
    T = ankle_z.shape[0]
    if T < max(3, int(round(fps * 0.4))) + 1:
        return float("nan"), float("nan")
    z = ankle_z.copy()
    if (~valid).any():
        idx = np.arange(T)
        if valid.any():
            z[~valid] = np.interp(idx[~valid], idx[valid], z[valid])
        else:
            return float("nan"), float("nan")
    k = max(3, int(round(fps / 10)) | 1)
    smoothed = np.convolve(z, np.ones(k) / k, mode="same")
    rng = float(smoothed.max() - smoothed.min())
    if rng <= 0:
        return float("nan"), float("nan")
    min_dist = max(3, int(round(fps * 0.4)))
    peaks, _ = find_peaks(smoothed, distance=min_dist,
                           prominence=0.10 * rng)
    if len(peaks) < 2:
        return float("nan"), float("nan")
    stride_s = float(np.median(np.diff(peaks)) / fps)
    if stride_s <= 0:
        return float("nan"), float("nan")
    return 60.0 / stride_s, stride_s


def _draw_gait_hud(
    img: np.ndarray,
    cad_l: float, cad_r: float,
    str_l: float, str_r: float,
) -> np.ndarray:
    """Bottom-of-panel translucent HUD with cadence + stride per side."""
    h, w = img.shape[:2]
    hud_h = 64
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - hud_h), (w, h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.62, img, 0.38, 0)

    def fmt_int(v: float) -> str:
        return "—" if not np.isfinite(v) else f"{v:>4.0f}"

    def fmt_f(v: float) -> str:
        return "—" if not np.isfinite(v) else f"{v:>4.2f}"

    line1 = f"cadence  L {fmt_int(cad_l)}  R {fmt_int(cad_r)}  steps/min"
    line2 = f"stride   L {fmt_f(str_l)}  R {fmt_f(str_r)}  s"
    cv2.putText(out, line1, (8, h - hud_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, line2, (8, h - hud_h + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


@app.command()
def main(
    session: str = typer.Option(..., "--session", help="e.g. S01/3"),
    run_dir: Path = typer.Option(Path("runs/run3_anatomical"), "--run-dir"),
    checkpoint: str = typer.Option("best", "--checkpoint"),
    out_dir: Path = typer.Option(Path("reports/videos"), "--out-dir"),
    fps: float = typer.Option(30.0, "--fps"),
    max_frames: int = typer.Option(0, "--max-frames", help="0 = all frames"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                        format="%(levelname)s %(message)s")
    cfg = Config.from_yaml(run_dir / "config.yaml")
    assert cfg.model is not None and cfg.training is not None
    subject, sess = session.split("/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load session frames & oracle parquet
    raw_dir = cfg.data.raw_dir
    base = WalkerSession(raw_dir, sessions=[(subject, sess)])
    n_frames = len(base) if max_frames <= 0 else min(max_frames, len(base))
    pq = pd.read_parquet(cfg.data.labels_dir / subject / f"{sess}.parquet")
    meta = json.loads((raw_dir / subject / sess / "meta.json").read_text())
    depth_intr = meta["depth_intrinsics"]

    # 2. Build the student dataset (returns model-ready tensors)
    ds = TrainingSession(
        raw_dir=raw_dir, labels_root=cfg.data.labels_dir,
        image_size=cfg.data.image_size,
        sessions=[(subject, sess)], split_file=None, split="all",
        drop_swaps=False,
    )
    # Index ds entries by frame_index for direct lookup
    fi_to_ds = {ds._items[i][3]: i for i in range(len(ds))}

    # 3. Build student
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthPoseStudent(
        backbone_name=cfg.model.backbone, num_joints=cfg.model.num_joints,
        num_deconv=cfg.model.num_deconv, deconv_channels=cfg.model.deconv_channels,
        softargmax_beta=cfg.model.softargmax_beta, pretrained=False,
    ).to(device)
    state = torch.load(run_dir / f"{checkpoint}.pt", map_location=device,
                       weights_only=True)
    model.load_state_dict(state["model"]); model.eval()
    logger.info("loaded student from %s/%s.pt", run_dir, checkpoint)

    # 4. Set up writer
    sample0 = base[0]
    H, W = sample0["depth_mm"].shape
    panel_h, panel_w = H, W
    # Caption strip 30 px tall
    cap_h = 30
    out_h = panel_h + cap_h
    out_w = panel_w * 2
    out_path = out_dir / f"{subject}_{sess}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open {out_path} for writing")

    logger.info("rendering %d frames → %s (%dx%d @ %.1f fps)",
                n_frames, out_path, out_w, out_h, fps)

    # ---- Pre-pass: build per-frame ankle-z series for both pipelines.
    # The HUD shows the *causal* running cadence/stride, computed from
    # ankle_z[0:k+1] at each render frame k.
    LH_i, RH_i = JOINT_ORDER.index("left_hip"),   JOINT_ORDER.index("right_hip")
    LK_i, RK_i = JOINT_ORDER.index("left_knee"),  JOINT_ORDER.index("right_knee")
    LA_i, RA_i = JOINT_ORDER.index("left_ankle"), JOINT_ORDER.index("right_ankle")

    # Map (subject, sess, fi) → ds_idx
    ds_idx_for_fi: dict[int, int] = fi_to_ds

    # Walk the base session in render order to capture frame_indices.
    frame_indices = [base[k]["frame_index"] for k in range(n_frames)]

    # ---- Oracle ankle_z from parquet (z_m on rows where depth_valid)
    pq_idx = pq.set_index(["frame_index", "joint_name"])
    oracle_zL = np.full(n_frames, np.nan, dtype=float)
    oracle_zR = np.full(n_frames, np.nan, dtype=float)
    oracle_vL = np.zeros(n_frames, dtype=bool)
    oracle_vR = np.zeros(n_frames, dtype=bool)
    for k, fi in enumerate(frame_indices):
        for jn, zarr, varr in [
            ("left_ankle",  oracle_zL, oracle_vL),
            ("right_ankle", oracle_zR, oracle_vR),
        ]:
            try:
                row = pq_idx.loc[(fi, jn)]
            except KeyError:
                continue
            if bool(row["depth_valid"]):
                zarr[k] = float(row["z_m"])
                varr[k] = True

    # ---- Student ankle_z by running model on every frame
    student_zL = np.zeros(n_frames, dtype=float)
    student_zR = np.zeros(n_frames, dtype=float)
    student_coords_cache: dict[int, np.ndarray] = {}
    logger.info("pre-computing student outputs for %d frames…", n_frames)
    with torch.inference_mode():
        for k, fi in enumerate(frame_indices):
            ds_idx = ds_idx_for_fi.get(fi)
            if ds_idx is None:
                continue
            item = ds[ds_idx]
            depth_t = item["depth"].unsqueeze(0).to(device)
            intr_t = item["intrinsics_input"].unsqueeze(0).to(device)
            coords = model(depth_t, intr_t)["coords_3d"][0].cpu().numpy()
            student_coords_cache[fi] = coords
            student_zL[k] = float(coords[LA_i, 2])
            student_zR[k] = float(coords[RA_i, 2])
    student_v = np.ones(n_frames, dtype=bool)  # student always emits a z

    # 5. Render loop
    for k in range(n_frames):
        sample = base[k]
        fi = sample["frame_index"]
        rgb = sample["rgb"]
        depth_mm = sample["depth_mm"]
        depth_color = _depth_colormap(depth_mm)

        # ---- Oracle 2D from parquet
        rows = pq[pq["frame_index"] == fi]
        oracle_uv: dict[str, tuple[int, int]] = {}
        oracle_valid: dict[str, bool] = {}
        for _, r in rows.iterrows():
            n = r["joint_name"]
            if n not in JOINT_ORDER:  # ignore non-target joints
                continue
            if r["conf_2d"] < 0.05:
                continue
            u, v = int(round(r["u_px"])), int(round(r["v_px"]))
            oracle_uv[n] = (u, v)
            oracle_valid[n] = bool(r["depth_valid"])

        # ---- Student 3D from cache → projected 2D in saved depth space
        if fi in student_coords_cache:
            coords_3d = student_coords_cache[fi]
            student_uv, student_valid = _project_to_depth(coords_3d, depth_intr)
        else:
            student_uv, student_valid = {}, {}

        # ---- Causal running cadence + stride (per pipeline, per side)
        sl = slice(0, k + 1)
        oc_cad_L, oc_str_L = _running_cadence_stride(
            oracle_zL[sl], oracle_vL[sl], fps)
        oc_cad_R, oc_str_R = _running_cadence_stride(
            oracle_zR[sl], oracle_vR[sl], fps)
        st_cad_L, st_str_L = _running_cadence_stride(
            student_zL[sl], student_v[sl], fps)
        st_cad_R, st_str_R = _running_cadence_stride(
            student_zR[sl], student_v[sl], fps)

        # ---- Compose
        rgb_overlay = _draw_skeleton(
            rgb, oracle_uv, oracle_valid,
            c_left=_ORACLE_LEFT, c_right=_ORACLE_RIGHT, c_line=_ORACLE_LINE,
        )
        depth_overlay = _draw_skeleton(
            depth_color, student_uv, student_valid,
            c_left=_STUDENT_LEFT, c_right=_STUDENT_RIGHT, c_line=_STUDENT_LINE,
        )
        # Live HUD on each panel
        rgb_overlay = _draw_gait_hud(rgb_overlay, oc_cad_L, oc_cad_R,
                                      oc_str_L, oc_str_R)
        depth_overlay = _draw_gait_hud(depth_overlay, st_cad_L, st_cad_R,
                                        st_str_L, st_str_R)

        # Captions on each panel
        rgb_cap = _strip_caption(panel_w, cap_h,
            f"Oracle (ViTPose++ on RGB)  -  {subject}/{sess}  f{fi:04d}")
        depth_cap = _strip_caption(panel_w, cap_h,
            "Student (depth-only, 5.2M params, 24ms@CPU)")
        # Highlight separator
        cv2.line(depth_overlay, (0, 0), (0, panel_h), (255, 255, 255), 1)
        top = np.concatenate([rgb_cap, depth_cap], axis=1)
        bottom = np.concatenate([rgb_overlay, depth_overlay], axis=1)
        out_frame = np.concatenate([top, bottom], axis=0)
        writer.write(out_frame)

        if (k + 1) % 50 == 0 or k + 1 == n_frames:
            logger.info("  %d/%d", k + 1, n_frames)

    writer.release()
    typer.echo(f"\nwrote {out_path}  (~{out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    app()
