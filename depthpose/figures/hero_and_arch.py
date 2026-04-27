"""Hero figure (one frame, both methods overlaid) and a method/architecture diagram.

Outputs:
- ``reports/figures/hero_frame.png``  — the single eye-catcher: RGB+oracle and
  depth+student side by side, plus a small text panel of headline numbers.
- ``reports/figures/oracle_student_training_diagram.png`` — a self-contained
  diagram of the pipeline: build-time oracle (RGB → ViTPose++ → median z →
  unproject) writes per-frame labels to a parquet on disk; inference-time
  student (depth → MobileNetV2 → 2.5D head → unproject) emits the deployed
  3D prediction. At train-time only, the loss reads labels from the parquet
  and compares against the student's predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer

from depthpose.data.dataset import WalkerSession
from depthpose.data.training_dataset import JOINT_ORDER, TrainingSession
from depthpose.figures.render_video import (
    _ORACLE_LEFT, _ORACLE_LINE, _ORACLE_RIGHT,
    _STUDENT_LEFT, _STUDENT_LINE, _STUDENT_RIGHT,
    _depth_colormap, _draw_skeleton, _project_to_depth,
)
from depthpose.model.student import DepthPoseStudent
from depthpose.training.config import Config

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

app = typer.Typer(add_completion=False, help=__doc__)


def _hero(session: str = "S01/14", frame_index: int = 195) -> Path:
    """Pick a representative frame and render the eye-catcher.

    Defaults: S01/14 (= rs_up_incline.bag) frame 195. This is a held-out
    bag the run4 model never saw during training, so the rendered student
    skeleton is a true generalisation result, not a memorised one.
    """
    cfg = Config.from_yaml(ROOT / "runs" / "run4_holdout_s01_14" / "config.yaml")
    subject, sess = session.split("/")

    base = WalkerSession(cfg.data.raw_dir, sessions=[(subject, sess)])
    # Find the dataset entry matching frame_index
    target_idx = next(i for i in range(len(base))
                      if base._items[i][3] == frame_index)
    sample = base[target_idx]
    rgb = sample["rgb"]; depth_mm = sample["depth_mm"]
    fi = sample["frame_index"]

    pq = pd.read_parquet(cfg.data.labels_dir / subject / f"{sess}.parquet")
    rows = pq[pq["frame_index"] == fi]
    oracle_uv: dict[str, tuple[int, int]] = {}; oracle_valid: dict[str, bool] = {}
    for _, r in rows.iterrows():
        if r["joint_name"] not in JOINT_ORDER or r["conf_2d"] < 0.05:
            continue
        oracle_uv[r["joint_name"]] = (int(round(r["u_px"])), int(round(r["v_px"])))
        oracle_valid[r["joint_name"]] = bool(r["depth_valid"])

    # Student
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthPoseStudent(
        backbone_name=cfg.model.backbone, num_joints=cfg.model.num_joints,
        num_deconv=cfg.model.num_deconv, deconv_channels=cfg.model.deconv_channels,
        softargmax_beta=cfg.model.softargmax_beta, pretrained=False,
    ).to(device)
    state = torch.load(ROOT / "runs" / "run4_holdout_s01_14" / "best.pt",
                       map_location=device, weights_only=True)
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

    # Compose figure: two image panels side-by-side with single-line titles.
    # Headline numbers + the "held-out" framing live in the prose caption
    # right below the figure.
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11.5, 6.8), constrained_layout=True,
    )
    ax_l.imshow(cv2.cvtColor(rgb_overlay, cv2.COLOR_BGR2RGB))
    ax_l.set_title("Oracle: ViTPose++ on RGB  (125 M params)",
                    fontsize=12)
    ax_l.axis("off")

    ax_r.imshow(cv2.cvtColor(depth_overlay, cv2.COLOR_BGR2RGB))
    ax_r.set_title("Student: depth-only, 5.2 M params, 0.95 ms / frame on GPU",
                    fontsize=12)
    ax_r.axis("off")

    p = FIG_DIR / "hero_frame.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def _draw_cylinder(ax, x, y, w, h, *, fill, edge, lw=1.4) -> None:
    """Database / parquet-on-disk cylinder shape, drawn from primitives.

    The cylinder reads as a different visual category from the rectangular
    compute boxes — it conveys "stored data, not a process."
    """
    cap_h = h * 0.22
    # Body fill (no outline, sits behind the cap arcs)
    body = mpatches.Rectangle(
        (x, y + cap_h / 2), w, h - cap_h,
        facecolor=fill, edgecolor="none", zorder=1)
    ax.add_patch(body)
    # Side lines
    ax.plot([x, x],         [y + cap_h / 2, y + h - cap_h / 2],
            color=edge, lw=lw, zorder=2, solid_capstyle="butt")
    ax.plot([x + w, x + w], [y + cap_h / 2, y + h - cap_h / 2],
            color=edge, lw=lw, zorder=2, solid_capstyle="butt")
    # Top ellipse (full)
    top = mpatches.Ellipse(
        (x + w / 2, y + h - cap_h / 2), w, cap_h,
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3)
    ax.add_patch(top)
    # Bottom: only the visible front arc
    bottom = mpatches.Arc(
        (x + w / 2, y + cap_h / 2), w, cap_h, theta1=180, theta2=360,
        edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(bottom)


def _method_diagram() -> Path:
    """Self-contained pipeline diagram drawn with matplotlib boxes & arrows.

    Layout:
      • Top zone — BUILD-TIME oracle pipeline. Solid boxes are compute
        steps that run once per recording.
      • Middle — Parquet labels stored on disk, drawn as a cylinder to
        signal "stored data, not a process."
      • Bottom zone — INFERENCE (and training) student. Solid boxes are
        compute steps that run every frame on the device.
      • Bottom-right — Training loss panel with a dashed border, signalling
        it only fires at training time. Dashed arrows feed it labels (read
        from disk) and predictions (computed by the student) — they go
        dashed because the connection only exists during training.

    Every node, edge, and annotation in this figure maps to a real line of
    code in ``depthpose/{oracle,model,training}/``. The diagram is drawn in
    pure matplotlib so it regenerates with no external dependencies.
    """
    # ============= Canvas =============
    W, H = 14.0, 9.6
    fig, ax = plt.subplots(figsize=(W, H), constrained_layout=True)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # ============= Palette (semantic, muted) =============
    INK         = "#1c1c1c"
    NEUTRAL     = "#666666"
    O_FILL      = "#eef2f7"; O_EDGE = "#385a87"        # oracle compute
    S_FILL      = "#eef3ef"; S_EDGE = "#2f5d3f"        # student compute
    STORE_FILL  = "#fbf3e2"; STORE_EDGE = "#8a6a2a"    # parquet on disk
    LOSS_FILL   = "#fdf1ec"; LOSS_EDGE = "#9a3a1a"     # train-only loss panel
    BAND_O      = "#f9fafc"
    BAND_S      = "#f9fbf9"

    # ============= Background bands =============
    ax.add_patch(mpatches.Rectangle((0, 7.10), W, 2.45,
                                     facecolor=BAND_O, edgecolor="none", zorder=0))
    ax.add_patch(mpatches.Rectangle((0, 2.05), W, 2.45,
                                     facecolor=BAND_S, edgecolor="none", zorder=0))

    # ============= Section headers =============
    ax.text(0.30, 9.32, "BUILD-TIME",
            fontsize=11, fontweight="bold", color=O_EDGE)
    ax.text(2.05, 9.32,
            "·  Oracle pipeline runs once per recording. Output is written to disk and never re-runs.",
            fontsize=10, color=NEUTRAL)

    ax.text(0.30, 4.72, "INFERENCE",
            fontsize=11, fontweight="bold", color=S_EDGE)
    ax.text(1.85, 4.72,
            "·  Student runs every frame on the device.",
            fontsize=10, color=NEUTRAL)
    ax.text(0.30, 4.36,
            "At train time, predictions are routed to the loss (dashed); at inference time, the 3D prediction is the deployed output.",
            fontsize=9.5, color=NEUTRAL, style="italic")

    # ============= Drawing helpers =============
    def _box(x, y, w, h, text, *, face, edge, fontsize=9.5, weight="normal"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.09",
            facecolor=face, edgecolor=edge, linewidth=1.15, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=INK, zorder=3)

    def _arrow(x1, y1, x2, y2, *, color=NEUTRAL, lw=1.3, ls="solid"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=13, shrinkA=2, shrinkB=2,
                                    linestyle=ls))

    # ===================================================================
    # ORACLE ROW (top) — solid boxes, solid arrows. Runs once per recording.
    # ===================================================================
    nodes_o = [
        (0.40, 1.55, "RGB frame"),
        (2.20, 1.90, "ViTPose++\n(HF transformers)\n6 of COCO-17 joints"),
        (4.40, 1.55, "(u, v) + conf\nper joint"),
        (6.25, 1.85, "3×3 median\ndepth at (u, v)"),
        (8.40, 1.65, "Unproject in\ndepth camera frame"),
        (10.45, 1.55, "3D label\n(X, Y, Z) m"),
    ]
    y_o, h_o = 7.55, 1.10
    for x, w, text in nodes_o:
        _box(x, y_o, w, h_o, text, face=O_FILL, edge=O_EDGE)
    for i in range(len(nodes_o) - 1):
        x1 = nodes_o[i][0] + nodes_o[i][1]
        x2 = nodes_o[i + 1][0]
        _arrow(x1, y_o + h_o / 2, x2, y_o + h_o / 2)


    # ===================================================================
    # PARQUET CYLINDER — between phases, signals "stored data, not a process"
    # ===================================================================
    o_term_cx = nodes_o[-1][0] + nodes_o[-1][1] / 2
    pq_w, pq_h = 2.60, 1.00
    pq_x = o_term_cx - pq_w / 2
    pq_y = 5.45
    _draw_cylinder(ax, pq_x, pq_y, pq_w, pq_h,
                   fill=STORE_FILL, edge=STORE_EDGE, lw=1.4)
    ax.text(pq_x + pq_w / 2, pq_y + pq_h * 0.62,
            "Per-frame labels",
            ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=STORE_EDGE, zorder=4)
    ax.text(pq_x + pq_w / 2, pq_y + pq_h * 0.32,
            "stored on disk",
            ha="center", va="center", fontsize=8.5, color=INK,
            style="italic", zorder=4)

    # solid 'saved' arrow: oracle terminal ↓ to parquet
    _arrow(o_term_cx, y_o, o_term_cx, pq_y + pq_h, color=O_EDGE, lw=1.4)
    ax.text(o_term_cx + 0.20, (y_o + pq_y + pq_h) / 2,
            "saved", fontsize=9.0, color=O_EDGE, style="italic",
            va="center", ha="left", fontweight="bold")

    # ===================================================================
    # STUDENT ROW (mid-bottom) — solid boxes, solid arrows. Runs every frame.
    # ===================================================================
    nodes_s = [
        (0.40, 1.55, "Depth frame\n192 × 256\n(in_chans = 1)"),
        (2.20, 1.90, "MobileNetV2\nbackbone\n(stride 32)"),
        (4.40, 1.55, "3 deconv blocks\n(→ stride 4)"),
        (6.25, 1.85, "1×1 conv\n→ 2J channels\n(heatmap + z-offset)"),
        (8.40, 1.65, "Soft-argmax\n→ (u, v, z)\n(shared weights)"),
        (10.45, 1.55, "Unproject\n→ 3D pred\n(X, Y, Z) m"),
    ]
    y_s, h_s = 2.55, 1.10
    for x, w, text in nodes_s:
        _box(x, y_s, w, h_s, text, face=S_FILL, edge=S_EDGE)
    for i in range(len(nodes_s) - 1):
        x1 = nodes_s[i][0] + nodes_s[i][1]
        x2 = nodes_s[i + 1][0]
        _arrow(x1, y_s + h_s / 2, x2, y_s + h_s / 2)

    # ===================================================================
    # TRAINING LOSS — dashed border = "exists only at train-time"
    # ===================================================================
    loss_x, loss_y = 0.55, 0.30
    loss_w, loss_h = 12.90, 1.40
    ax.add_patch(mpatches.FancyBboxPatch(
        (loss_x, loss_y), loss_w, loss_h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=LOSS_FILL, edgecolor=LOSS_EDGE, linewidth=1.4, zorder=2,
        linestyle=(0, (5, 3))))
    ax.text(loss_x + 0.25, loss_y + loss_h - 0.22,
            "Training loss",
            ha="left", va="top", fontsize=10.5, fontweight="bold",
            color=LOSS_EDGE, zorder=3)
    ax.text(loss_x + 1.95, loss_y + loss_h - 0.24,
            "(train-time only — at inference, the student's 3D prediction is the deployed output)",
            ha="left", va="top", fontsize=9.0, color=NEUTRAL,
            style="italic", zorder=3)

    bullets = [
        ("Smooth-L1 on (X, Y, Z)",
         "per-joint mask:  depth_valid  &  ~needs_review"),
        ("aux 2D heatmap MSE   × 0.1",
         "Gaussian target at (u, v) / head_stride;  mask:  ~needs_review"),
        ("anatomical hinge   × 1.0",
         "mean of  hip×knee, knee×ankle, hip×ankle  ReLU pair hinges"),
    ]
    col_w = (loss_w - 0.40) / 3.0
    for i, (head, sub) in enumerate(bullets):
        cx = loss_x + 0.20 + col_w * i + col_w / 2
        ax.text(cx, loss_y + 0.65, "•  " + head,
                fontsize=9.5, ha="center", va="center",
                fontweight="bold", color=INK, zorder=3)
        ax.text(cx, loss_y + 0.30, sub,
                fontsize=8.5, ha="center", va="center", color="#555",
                zorder=3)

    # Dashed arrow: parquet → loss top  (train-time read of stored labels).
    # Single L-shaped arrow that exits the right side of the parquet,
    # runs down the right gutter (clear of the student boxes), and enters
    # the loss block from above-right.
    rail_x = pq_x + pq_w + 0.35   # ≈ 12.9, clear of student row's last box
    ax.annotate("",
                xy=(rail_x, loss_y + loss_h),
                xytext=(pq_x + pq_w, pq_y + pq_h * 0.5),
                arrowprops=dict(arrowstyle="-|>", color=STORE_EDGE, lw=1.4,
                                mutation_scale=14, shrinkA=0, shrinkB=2,
                                linestyle=(0, (4, 3)),
                                connectionstyle="angle,angleA=0,angleB=-90,rad=0"))
    ax.text(rail_x + 0.10, (pq_y + loss_y + loss_h) / 2,
            "labels\n(read at train)",
            fontsize=8.5, color=STORE_EDGE, style="italic",
            ha="left", va="center", fontweight="bold")

    # Dashed arrow: student 3D pred → loss top (computed at train-time).
    # Short, vertical-ish; enters the loss block at its top centre.
    s_term_cx = nodes_s[-1][0] + nodes_s[-1][1] / 2
    _arrow(s_term_cx, y_s, s_term_cx, loss_y + loss_h,
           color=S_EDGE, lw=1.3, ls=(0, (4, 3)))
    ax.text(s_term_cx + 0.18, (y_s + loss_y + loss_h) / 2,
            "predictions\n(train only)",
            fontsize=8.5, color=S_EDGE, style="italic",
            ha="left", va="center", fontweight="bold")

    # ============= Deployment badge =============
    # Light-touch text badge in the gap between the student row and the
    # loss block, left-aligned so it does not collide with the predictions
    # arrow that runs down on the right side.
    ax.text(0.55, loss_y + loss_h + 0.32,
            "Deployed:",
            fontsize=9.0, color=S_EDGE, ha="left", va="center",
            fontweight="bold", zorder=3)
    ax.text(2.05, loss_y + loss_h + 0.32,
            "5.22 M params  ·  19.9 MB ONNX  ·  0.95 ms GPU  ·  24 ms CPU 1-thread",
            fontsize=9.0, color=INK, ha="left", va="center", zorder=3)

    p = FIG_DIR / "oracle_student_training_diagram.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


@app.command()
def main(
    session: str = typer.Option("S01/14", "--session"),
    frame_index: int = typer.Option(195, "--frame-index"),
) -> None:
    p1 = _hero(session=session, frame_index=frame_index)
    p2 = _method_diagram()
    typer.echo(f"hero    {p1.relative_to(ROOT)}")
    typer.echo(f"method  {p2.relative_to(ROOT)}")


if __name__ == "__main__":
    app()
