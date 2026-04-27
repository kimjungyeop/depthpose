"""Visual + numerical validation of the oracle parquet.

Produces, per session:
- ``<session>_contact_sheet.png`` — 4×6 grid of evenly-spaced frames,
  each cell showing the RGB overlay (left half) and the depth
  colourmap with the same keypoints (right half).
- ``<session>_histograms.png`` — per-joint confidence histogram and
  per-joint Z (camera-frame depth) histogram.
- ``<session>_summary.md`` — counts and fractions: per-joint
  ``mean_conf``, ``frac_depth_valid``, ``frac_needs_review``,
  ``median_z``; plus the global all-joints-valid frame count.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from depthpose.data.dataset import WalkerSession
from depthpose.oracle.quality import is_frame_skeleton_consistent

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


_PALETTE_BGR: dict[str, tuple[int, int, int]] = {
    "left_hip":    (255, 64, 64),
    "right_hip":   (64, 64, 255),
    "left_knee":   (255, 128, 64),
    "right_knee":  (64, 128, 255),
    "left_ankle":  (255, 192, 64),
    "right_ankle": (64, 192, 255),
}
_FULL_SKELETON: list[tuple[str, str]] = [
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_hip", "right_hip"),
]
_DEPTH_VIS_LO_MM = 300
_DEPTH_VIS_HI_MM = 1800


def _depth_colormap(depth_mm: np.ndarray) -> np.ndarray:
    norm = np.clip(depth_mm, _DEPTH_VIS_LO_MM, _DEPTH_VIS_HI_MM).astype(np.float32)
    norm = ((norm - _DEPTH_VIS_LO_MM) /
            (_DEPTH_VIS_HI_MM - _DEPTH_VIS_LO_MM) * 255).astype(np.uint8)
    out = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    out[depth_mm == 0] = (0, 0, 0)
    return out


def _draw_keypoints(img: np.ndarray, frame_rows: pd.DataFrame, *,
                    score_thresh: float = 0.05) -> np.ndarray:
    out = img.copy()
    name_to_uv: dict[str, tuple[int, int]] = {}
    for _, row in frame_rows.iterrows():
        u, v = float(row["u_px"]), float(row["v_px"])
        if row["conf_2d"] < score_thresh:
            continue
        name_to_uv[row["joint_name"]] = (int(round(u)), int(round(v)))
    # skeleton first
    for a, b in _FULL_SKELETON:
        if a in name_to_uv and b in name_to_uv:
            cv2.line(out, name_to_uv[a], name_to_uv[b], (240, 240, 240), 2)
    # joints
    for _, row in frame_rows.iterrows():
        if row["conf_2d"] < score_thresh:
            continue
        u, v = int(round(row["u_px"])), int(round(row["v_px"]))
        c = _PALETTE_BGR.get(row["joint_name"], (0, 255, 0))
        # outer ring grey if depth invalid
        ring = (200, 200, 200) if row["depth_valid"] else (60, 60, 60)
        cv2.circle(out, (u, v), 7, ring, 2)
        cv2.circle(out, (u, v), 4, c, -1)
    return out


def _label(img: np.ndarray, text: str, *, scale: float = 0.5) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_contact_sheet(
    raw_dir: Path,
    subject: str,
    session: str,
    df: pd.DataFrame,
    out_path: Path,
    n_cells: int = 24,
    cols: int = 6,
) -> None:
    ds = WalkerSession(raw_dir, sessions=[(subject, session)])
    n = len(ds)
    if n_cells > n:
        n_cells = n
    indices = np.linspace(0, n - 1, n_cells, dtype=int).tolist()
    rows = (n_cells + cols - 1) // cols

    cells: list[np.ndarray] = []
    for ds_idx in indices:
        sample = ds[ds_idx]
        fi = sample["frame_index"]
        rgb = sample["rgb"]
        depth = sample["depth_mm"]
        rows_for_frame = df[df["frame_index"] == fi]
        rgb_overlay = _draw_keypoints(rgb, rows_for_frame)
        depth_color = _depth_colormap(depth)
        depth_overlay = _draw_keypoints(depth_color, rows_for_frame)
        # downscale for the grid
        H, W = rgb_overlay.shape[:2]
        scale = 0.42
        side = np.concatenate([rgb_overlay, depth_overlay], axis=1)
        side = cv2.resize(side, (int(side.shape[1] * scale),
                                  int(side.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)
        ka_rows = rows_for_frame[rows_for_frame["joint_name"].isin(_KNEE_ANKLE_JOINTS)]
        ka_ok = bool(ka_rows["depth_valid"].all())
        hip_rows = rows_for_frame[rows_for_frame["joint_name"].str.endswith("_hip")]
        hip_ok = bool(hip_rows["depth_valid"].all())
        skel_ok = is_frame_skeleton_consistent(rows_for_frame)
        if not skel_ok:
            flag = "X SWAP"
        elif not ka_ok:
            flag = "K/A MISS"
        elif hip_ok:
            flag = "all 6"
        else:
            flag = "no hip"
        side = _label(side, f"f{fi:04d}  {flag}")
        cells.append(side)

    cell_h, cell_w = cells[0].shape[:2]
    grid = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    for k, c in enumerate(cells):
        r, ccol = divmod(k, cols)
        grid[r * cell_h:(r + 1) * cell_h, ccol * cell_w:(ccol + 1) * cell_w] = c

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    logger.info("wrote contact sheet %s (%d cells, %d×%d cell)",
                out_path, n_cells, cell_w, cell_h)


def build_histograms(df: pd.DataFrame, out_path: Path) -> None:
    joints = sorted(df["joint_name"].unique())
    fig, axes = plt.subplots(2, len(joints), figsize=(2.4 * len(joints), 5),
                             constrained_layout=True)
    for i, joint in enumerate(joints):
        sub = df[df["joint_name"] == joint]
        axes[0, i].hist(sub["conf_2d"], bins=20, range=(0, 1), color="#377eb8")
        axes[0, i].set_title(f"{joint}\nconf_2d", fontsize=9)
        axes[0, i].set_xlim(0, 1)
        axes[0, i].set_ylabel("frames" if i == 0 else "")

        z_valid = sub.loc[sub["depth_valid"], "z_m"]
        axes[1, i].hist(z_valid, bins=30, color="#e41a1c")
        axes[1, i].set_title("z_m (valid)", fontsize=9)
        axes[1, i].set_xlabel("metres")
        axes[1, i].set_ylabel("frames" if i == 0 else "")

    fig.suptitle("Oracle per-joint histograms", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("wrote histograms %s", out_path)


_KNEE_ANKLE_JOINTS: tuple[str, ...] = (
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def _frac_frames_with_all_joints_valid(
    df: pd.DataFrame,
    joints: tuple[str, ...] | None = None,
) -> tuple[int, int]:
    """Return (n_valid_frames, n_total_frames) where every joint in ``joints``
    has both ``depth_valid`` and not ``needs_review``."""
    sub = df if joints is None else df[df["joint_name"].isin(joints)]
    valid_per_row = sub["depth_valid"] & ~sub["needs_review"]
    by_frame = (
        sub.assign(_v=valid_per_row)
        .groupby("frame_index")["_v"].all()
    )
    return int(by_frame.sum()), int(by_frame.size)


def build_summary_md(
    df: pd.DataFrame,
    subject: str,
    session: str,
    out_path: Path,
) -> None:
    n_frames = int(df["frame_index"].nunique())
    per_joint = (
        df.groupby("joint_name")
        .agg(
            mean_conf=("conf_2d", "mean"),
            frac_depth_valid=("depth_valid", "mean"),
            frac_needs_review=("needs_review", "mean"),
            median_z=("z_m", lambda s: float(s.median())),
        )
        .round(3)
    )

    n_all6, _ = _frac_frames_with_all_joints_valid(df, joints=None)
    n_kneeankle, _ = _frac_frames_with_all_joints_valid(df, joints=_KNEE_ANKLE_JOINTS)

    lines: list[str] = []
    lines.append(f"# Oracle validation summary — {subject}/{session}")
    lines.append("")
    lines.append(f"- frames: **{n_frames}**")
    lines.append("")
    lines.append("## Frame-eligibility metrics")
    lines.append("")
    lines.append("| metric | count | fraction |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| all 6 joints valid (strict) | {n_all6} / {n_frames} | "
        f"{n_all6 / n_frames:.3f} |"
    )
    lines.append(
        f"| knees + ankles valid (gait-relevant) | "
        f"{n_kneeankle} / {n_frames} | {n_kneeankle / n_frames:.3f} |"
    )
    lines.append("")
    lines.append("**Use `knees + ankles valid` as the frame-eligibility metric.** "
                 "Hip drop-outs are a known depth-FOV-gap property and should "
                 "not disqualify the rest of the frame. Per-joint masking — "
                 "*never* whole-frame filtering — is the policy for both "
                 "training loss and evaluation MPJPE (see project memory: "
                 "per-joint masking).")
    lines.append("")
    lines.append("## Per-joint")
    lines.append("")
    lines.append("| joint | mean conf | frac depth valid | frac needs review | median z (m) |")
    lines.append("|---|---:|---:|---:|---:|")
    for joint, row in per_joint.iterrows():
        lines.append(
            f"| {joint} | {row['mean_conf']:.3f} | "
            f"{row['frac_depth_valid']:.3f} | "
            f"{row['frac_needs_review']:.3f} | "
            f"{row['median_z']:.3f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `conf_2d` is the ViTPose detection score in [0, 1].")
    lines.append("- `depth_valid` = depth-median sampling returned a "
                 "nonzero value (3×3 patch).")
    lines.append("- `needs_review` = `conf_2d` < threshold (default 0.5).")
    lines.append("- Hip frames where `depth_valid=False` are typically "
                 "the stride-extension instants when the hip moves into "
                 "the depth-FOV gap at the top of the frame "
                 "(see project memory: hip-depth-invalid policy).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    logger.info("wrote summary %s", out_path)


@app.command()
def main(
    session: Path = typer.Option(..., "--session",
                                  help="Path like data/raw/<subject>/<session>"),
    labels: Path = typer.Option(..., "--labels"),
    out_dir: Path = typer.Option(Path("reports/oracle_validate"), "--out-dir"),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    n_cells: int = typer.Option(24, "--n-cells"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Build contact sheet + histograms + summary md for one session."""
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                        format="%(levelname)s %(message)s")
    if session.parent.parent != raw_dir:
        # Allow looser matching but warn — accept any session dir.
        logger.warning("session=%s is not under raw_dir=%s; continuing anyway",
                       session, raw_dir)
    subject = session.parent.name
    session_id = session.name
    df = pd.read_parquet(labels)
    df = df[(df["subject"] == subject) & (df["session"] == session_id)]
    if df.empty:
        raise typer.BadParameter(
            f"no rows for {subject}/{session_id} in {labels}"
        )

    stem = f"{subject}_{session_id}"
    build_contact_sheet(
        raw_dir=raw_dir, subject=subject, session=session_id, df=df,
        out_path=out_dir / f"{stem}_contact_sheet.png",
        n_cells=n_cells,
    )
    build_histograms(df, out_dir / f"{stem}_histograms.png")
    build_summary_md(df, subject, session_id, out_dir / f"{stem}_summary.md")
    typer.echo(f"validation artefacts written to {out_dir}/")


if __name__ == "__main__":
    app()
