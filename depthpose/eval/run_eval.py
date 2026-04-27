"""Run a trained checkpoint over the test split and produce a structured report.

Outputs into the run's directory:
- ``eval.json``           machine-readable: per-joint MPJPE, PCK@thresholds, etc.
- ``eval_report.md``      human-readable markdown summary.
- ``pck_curve.png``       PCK at thresholds 0..100 mm (overall + per-joint).
- ``per_joint_error.png`` per-joint MPJPE bar chart.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from depthpose.data.training_dataset import JOINT_ORDER, TrainingSession
from depthpose.eval.metrics import (
    mpjpe_overall_mm,
    mpjpe_per_joint_mm,
    pck_curve,
    pck_overall,
    pck_per_joint,
)
from depthpose.model.student import DepthPoseStudent
from depthpose.training.config import Config

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on loader; return (pred (N,J,3), target (N,J,3), valid (N,J))."""
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        depth = batch["depth"].to(device, non_blocking=True)
        intr = batch["intrinsics_input"].to(device, non_blocking=True)
        pred = model(depth, intr)
        preds.append(pred["coords_3d"].cpu().numpy())
        targets.append(batch["target_3d"].numpy())
        valid = (batch["depth_valid"] & ~batch["needs_review"]).numpy()
        valids.append(valid)
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(valids, axis=0),
    )


@app.command()
def main(
    run_dir: Path = typer.Option(..., "--run-dir"),
    checkpoint: str = typer.Option("best", "--checkpoint", help="best | last"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.from_yaml(run_dir / "config.yaml")
    assert cfg.model is not None and cfg.training is not None and cfg.eval is not None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = TrainingSession(
        raw_dir=cfg.data.raw_dir,
        labels_root=cfg.data.labels_dir,
        image_size=cfg.data.image_size,
        split_file=cfg.data.splits_path,
        split="test",
        drop_swaps=True,
    )
    loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.training.num_workers, pin_memory=True,
    )

    model = DepthPoseStudent(
        backbone_name=cfg.model.backbone,
        num_joints=cfg.model.num_joints,
        num_deconv=cfg.model.num_deconv,
        deconv_channels=cfg.model.deconv_channels,
        softargmax_beta=cfg.model.softargmax_beta,
        pretrained=False,
    ).to(device)
    ckpt_path = run_dir / f"{checkpoint}.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    logger.info("loaded %s (epoch %d)", ckpt_path, state.get("epoch", -1))

    pred, target, valid = collect_predictions(model, loader, device)
    logger.info("predictions: %s frames, %s joints, %d valid total",
                pred.shape[0], pred.shape[1], int(valid.sum()))

    overall_mpjpe = mpjpe_overall_mm(pred, target, valid)
    per_joint = mpjpe_per_joint_mm(pred, target, valid)
    pck_o = pck_overall(pred, target, valid, cfg.eval.pck_thresholds_mm)
    pck_j = pck_per_joint(pred, target, valid, cfg.eval.pck_thresholds_mm)
    ts, pck_curve_y = pck_curve(pred, target, valid, max_mm=100, step_mm=1)

    eval_payload = {
        "checkpoint": str(ckpt_path),
        "epoch": int(state.get("epoch", -1)),
        "n_frames": int(pred.shape[0]),
        "n_valid_joints": int(valid.sum()),
        "joints": list(JOINT_ORDER),
        "mpjpe_mm_overall": overall_mpjpe,
        "mpjpe_mm_per_joint": {j: float(per_joint[i]) for i, j in enumerate(JOINT_ORDER)},
        "pck_overall": {f"{t}mm": pck_o[t] for t in cfg.eval.pck_thresholds_mm},
        "pck_per_joint": {
            f"{t}mm": {j: float(pck_j[t][i]) for i, j in enumerate(JOINT_ORDER)}
            for t in cfg.eval.pck_thresholds_mm
        },
        "pck_curve": {"thresholds_mm": ts.tolist(), "pck": pck_curve_y.tolist()},
    }
    (run_dir / "eval.json").write_text(json.dumps(eval_payload, indent=2))

    # ---- markdown ----
    lines: list[str] = []
    lines.append(f"# Evaluation report — `{run_dir.name}`")
    lines.append("")
    lines.append(f"- checkpoint: `{ckpt_path.name}` (epoch {state.get('epoch','?')})")
    lines.append(f"- frames evaluated: **{pred.shape[0]}**, "
                 f"valid joint preds: **{int(valid.sum())}**")
    lines.append(f"- **MPJPE overall: {overall_mpjpe:.1f} mm**  "
                 f"(brief target H1: <35 mm)")
    lines.append("")
    lines.append("## Per-joint MPJPE (mm)")
    lines.append("")
    lines.append("| joint | MPJPE (mm) |")
    lines.append("|---|---:|")
    for i, j in enumerate(JOINT_ORDER):
        v = per_joint[i]
        cell = "—" if np.isnan(v) else f"{v:.1f}"
        lines.append(f"| {j} | {cell} |")
    lines.append("")
    lines.append("## PCK")
    lines.append("")
    cols = " | ".join(f"@{t}mm" for t in cfg.eval.pck_thresholds_mm)
    lines.append(f"| joint | {cols} |")
    lines.append("|---|" + "---:|" * len(cfg.eval.pck_thresholds_mm))
    for i, j in enumerate(JOINT_ORDER):
        cells = " | ".join(f"{pck_j[t][i]*100:.1f}%" for t in cfg.eval.pck_thresholds_mm)
        lines.append(f"| {j} | {cells} |")
    overall_cells = " | ".join(f"{pck_o[t]*100:.1f}%" for t in cfg.eval.pck_thresholds_mm)
    lines.append(f"| **overall** | {overall_cells} |")
    (run_dir / "eval_report.md").write_text("\n".join(lines))

    # ---- PCK curve plot ----
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(ts, pck_curve_y, "k-", lw=2, label="overall (all joints)")
    palette = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    for i, j in enumerate(JOINT_ORDER):
        ts_j, pck_j_curve = pck_curve(
            pred[:, i:i + 1], target[:, i:i + 1], valid[:, i:i + 1],
            max_mm=100, step_mm=1,
        )
        ax.plot(ts_j, pck_j_curve, lw=1.0, color=palette[i % len(palette)],
                label=j, alpha=0.85)
    for t in cfg.eval.pck_thresholds_mm:
        ax.axvline(t, color="grey", lw=0.5, alpha=0.4)
    ax.set_xlabel("threshold (mm)")
    ax.set_ylabel("PCK")
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.0)
    ax.set_title(f"PCK — {run_dir.name} (overall MPJPE {overall_mpjpe:.1f} mm)")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(run_dir / "pck_curve.png", dpi=120)
    plt.close(fig)

    # ---- per-joint MPJPE bar ----
    fig2, ax2 = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    valid_pj = ~np.isnan(per_joint)
    ax2.bar(np.arange(len(JOINT_ORDER))[valid_pj], per_joint[valid_pj],
            color=[palette[i % len(palette)] for i in range(len(JOINT_ORDER)) if valid_pj[i]])
    ax2.axhline(35, color="red", lw=1, ls="--", label="H1 target 35 mm")
    ax2.set_xticks(range(len(JOINT_ORDER)))
    ax2.set_xticklabels(JOINT_ORDER, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("MPJPE (mm)")
    ax2.set_title(f"Per-joint MPJPE — {run_dir.name}")
    ax2.legend()
    fig2.savefig(run_dir / "per_joint_error.png", dpi=120)
    plt.close(fig2)

    typer.echo("\n=== eval summary ===")
    typer.echo(f"frames: {pred.shape[0]}, valid: {int(valid.sum())}")
    typer.echo(f"MPJPE overall: {overall_mpjpe:.1f} mm  (H1 target <35 mm)")
    for i, j in enumerate(JOINT_ORDER):
        v = per_joint[i]
        cell = "—" if np.isnan(v) else f"{v:.1f}"
        typer.echo(f"  {j:<14} {cell} mm")
    for t in cfg.eval.pck_thresholds_mm:
        typer.echo(f"PCK@{t}mm overall: {pck_o[t]*100:.1f}%")


if __name__ == "__main__":
    app()
