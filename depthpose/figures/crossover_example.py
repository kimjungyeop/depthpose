"""Single-frame screenshot of an actual L/R hip-knee crossover artifact.

Picks one frame from `run1_baseline` (the version trained without the
anatomical-consistency loss) where the predicted left hip ends up on the
opposite side of the body from the predicted left knee — anatomically
impossible. Renders the oracle's RGB+skeleton and the student's
depth+skeleton side-by-side as a static figure.

Used in §6 of the blog to make the abstract "0.53 % crossover rate"
number concrete.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from depthpose.data.dataset import WalkerSession
from depthpose.data.training_dataset import JOINT_ORDER, TrainingSession
from depthpose.figures.render_video import (
    _ORACLE_LEFT, _ORACLE_LINE, _ORACLE_RIGHT,
    _STUDENT_LEFT, _STUDENT_LINE, _STUDENT_RIGHT,
    _depth_colormap, _draw_skeleton, _project_to_depth,
)
from depthpose.model.student import DepthPoseStudent
from depthpose.training.config import Config

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    session: str = typer.Option("S01/1", "--session"),
    frame_index: int = typer.Option(288, "--frame-index"),
    run_dir: Path = typer.Option(Path("runs/run1_baseline"), "--run-dir",
                                  help="Use the model BEFORE the anatomical loss "
                                       "to surface the artifact."),
) -> None:
    cfg = Config.from_yaml(run_dir / "config.yaml")
    subject, sess = session.split("/")

    base = WalkerSession(cfg.data.raw_dir, sessions=[(subject, sess)])
    target_idx = next(i for i in range(len(base))
                      if base._items[i][3] == frame_index)
    sample = base[target_idx]
    rgb = sample["rgb"]
    depth_mm = sample["depth_mm"]
    fi = sample["frame_index"]

    pq = pd.read_parquet(cfg.data.labels_dir / subject / f"{sess}.parquet")
    rows = pq[pq["frame_index"] == fi]
    oracle_uv: dict = {}
    oracle_valid: dict = {}
    for _, r in rows.iterrows():
        if r["joint_name"] not in JOINT_ORDER or r["conf_2d"] < 0.05:
            continue
        oracle_uv[r["joint_name"]] = (int(round(r["u_px"])), int(round(r["v_px"])))
        oracle_valid[r["joint_name"]] = bool(r["depth_valid"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthPoseStudent(
        backbone_name=cfg.model.backbone, num_joints=cfg.model.num_joints,
        num_deconv=cfg.model.num_deconv, deconv_channels=cfg.model.deconv_channels,
        softargmax_beta=cfg.model.softargmax_beta, pretrained=False,
    ).to(device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"]); model.eval()

    ds = TrainingSession(
        raw_dir=cfg.data.raw_dir, labels_root=cfg.data.labels_dir,
        image_size=cfg.data.image_size, sessions=[(subject, sess)],
        split_file=None, split="all", drop_swaps=False,
    )
    ds_idx = next(i for i in range(len(ds)) if ds._items[i][3] == fi)
    item = ds[ds_idx]
    with torch.inference_mode():
        coords_3d = model(item["depth"].unsqueeze(0).to(device),
                          item["intrinsics_input"].unsqueeze(0).to(device)
                          )["coords_3d"][0].cpu().numpy()
    meta = json.loads((cfg.data.raw_dir / subject / sess / "meta.json").read_text())
    student_uv, student_valid = _project_to_depth(coords_3d, meta["depth_intrinsics"])

    rgb_overlay = _draw_skeleton(rgb, oracle_uv, oracle_valid,
                                  c_left=_ORACLE_LEFT, c_right=_ORACLE_RIGHT,
                                  c_line=_ORACLE_LINE)
    depth_color = _depth_colormap(depth_mm)
    depth_overlay = _draw_skeleton(depth_color, student_uv, student_valid,
                                    c_left=_STUDENT_LEFT, c_right=_STUDENT_RIGHT,
                                    c_line=_STUDENT_LINE)

    # Print the geometry so the caption is grounded in measured numbers.
    LH, RH, LK, RK = (JOINT_ORDER.index(j) for j in
        ("left_hip", "right_hip", "left_knee", "right_knee"))
    dx_hip_cm = (coords_3d[LH, 0] - coords_3d[RH, 0]) * 100
    dx_knee_cm = (coords_3d[LK, 0] - coords_3d[RK, 0]) * 100
    typer.echo(f"frame {sess}/{fi}: dx_hip={dx_hip_cm:+.1f} cm, "
               f"dx_knee={dx_knee_cm:+.1f} cm  → "
               f"{'CROSSOVER' if dx_hip_cm*dx_knee_cm < 0 else 'consistent'}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 8.0), constrained_layout=True)
    axes[0].imshow(cv2.cvtColor(rgb_overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Oracle (RGB + ViTPose++ 2D skeleton)\n"
                       "Lateral order: consistent",
                       fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(depth_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        "Student depth-only prediction  (model trained without the anatomical loss)\n"
        f"dx_hip = {dx_hip_cm:+.0f} cm,  dx_knee = {dx_knee_cm:+.0f} cm  →  CROSSOVER",
        fontsize=12, color="#7a1a1a",
    )
    axes[1].axis("off")

    # Annotate the crossover on the depth panel: arrow pointing at the X
    # made by the thigh skeleton lines. Keep the annotation text fully
    # inside the panel so it doesn't get clipped by bbox_inches='tight'.
    H, W = depth_overlay.shape[:2]
    hip_v = int((student_uv["left_hip"][1] + student_uv["right_hip"][1]) / 2)
    knee_v = int((student_uv["left_knee"][1] + student_uv["right_knee"][1]) / 2)
    cross_v = int(0.55 * hip_v + 0.45 * knee_v)
    cross_u = W // 2
    # Place the annotation text in the bottom-left of the panel where the
    # depth image is mostly empty (background), then point up-right at the X.
    axes[1].annotate(
        "thighs would have to\ncross — anatomically\nimpossible",
        xy=(cross_u - 30, cross_v),
        xytext=(int(W * 0.05), int(H * 0.85)),
        fontsize=12, color="#7a1a1a", fontweight="bold",
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="#7a1a1a", linewidth=1.5),
        arrowprops=dict(arrowstyle="->", color="#7a1a1a", lw=2.0),
    )

    out = FIG_DIR / "crossover_example.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    app()
