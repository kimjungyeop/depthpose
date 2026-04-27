"""Generate the figures the blog post will reference.

Outputs to ``reports/figures/``:
- ``per_session_mpjpe.png`` — bar chart of per-session MPJPE on the test split,
  with the H1 target line and outlier highlighting.
- ``latency_comparison.png`` — bar chart of inference latency across runtimes.
- ``axis_breakdown.png`` — stacked-bar of per-axis error per session, showing
  the distinct failure modes of S01/3 (z-dominated) vs S01/5 (x-dominated).

Idempotent — overwrite existing PNGs. Reads only from JSON sidecars produced
by earlier phases, so no model loading is needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

app = typer.Typer(add_completion=False, help=__doc__)


def _per_session_mpjpe_bar() -> Path:
    """Bar chart of per-session test-set MPJPE."""
    data = json.loads((ROOT / "reports" / "per_session_mpjpe.json").read_text())
    sessions = sorted(data.keys(), key=lambda k: int(k.split("/")[1]))
    mpjpes = [data[s]["mpjpe_overall_mm"] for s in sessions]
    n_frames = [data[s]["n_frames"] for s in sessions]

    # Highlight outliers in a different colour.
    palette = []
    for s, m in zip(sessions, mpjpes):
        if m > 35:
            palette.append("#d62728")  # red — over H1 target
        elif m > 25:
            palette.append("#ff7f0e")  # orange — high but under target
        else:
            palette.append("#2ca02c")  # green

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x = np.arange(len(sessions))
    bars = ax.bar(x, mpjpes, color=palette, edgecolor="k", linewidth=0.5)
    for b, n, v in zip(bars, n_frames, mpjpes):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(35, color="red", lw=1.2, ls="--",
               label="H1 target: <35 mm MPJPE")
    avg_all = float(np.mean(mpjpes))
    ax.axhline(avg_all, color="grey", lw=1.0, ls=":",
               label=f"average over all sessions: {avg_all:.1f} mm")
    avg_no_outliers = float(np.mean([m for m in mpjpes if m < 35]))
    ax.axhline(avg_no_outliers, color="black", lw=1.0, ls=":",
               label=f"average excl. outliers: {avg_no_outliers:.1f} mm")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s + f"\n(n={n})" for s, n in zip(sessions, n_frames)],
        rotation=0, fontsize=7,
    )
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Per-session test-set MPJPE — depth-only student vs ViTPose++ oracle")
    ax.set_ylim(0, max(mpjpes) * 1.15)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

    p = FIG_DIR / "per_session_mpjpe.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _latency_bar() -> Path:
    """Bar chart comparing inference latency across runtimes."""
    bench = json.loads(
        (ROOT / "runs" / "run1_baseline" / "latency_benchmark.json").read_text()
    )
    res = bench["results"]
    res_gpu = bench.get("results_gpu", {})
    # Order: slow-CPU → fast-CPU → GPU.
    order = [
        ("ONNX-RT CPU 1 thread",  "onnxruntime_cpu_1thread", "#ff7f0e", res),
        ("PyTorch CPU",           "pytorch_cpu", "#1f77b4", res),
        ("ONNX-RT CPU (multi)",   "onnxruntime_cpu_default_threads", "#2ca02c", res),
        ("PyTorch CUDA",          "pytorch_cuda", "#9467bd", res),
        ("ONNX-RT CUDA",          "onnxruntime_cuda_fp32", "#e377c2", res_gpu),
    ]
    # Filter to runtimes actually present.
    order = [o for o in order if o[1] in o[3] and "median_ms" in o[3][o[1]]]
    labels = [o[0] for o in order]
    medians = [o[3][o[1]]["median_ms"] for o in order]
    p95s = [o[3][o[1]]["p95_ms"] for o in order]
    colors = [o[2] for o in order]

    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, medians, color=colors, edgecolor="k", linewidth=0.5)
    # Whiskers to p95.
    ax.errorbar(x, medians,
                yerr=[[0]*len(medians), [p - m for p, m in zip(p95s, medians)]],
                fmt="none", color="black", capsize=4, linewidth=1.0)
    for b, m in zip(bars, medians):
        fps = 1000.0 / m
        ax.text(b.get_x() + b.get_width() / 2, m + 0.4,
                f"{m:.1f} ms\n({fps:.0f} fps)",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(33.3, color="red", lw=1.2, ls="--",
               label="30 fps real-time budget (33.3 ms)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("inference latency at batch 1 (ms, lower is better)")
    ax.set_title(f"Student inference latency — {bench['system'].get('cuda_device') or 'no CUDA'}\n"
                 f"({bench['system']['cpu_count']} CPU cores)")
    ax.set_ylim(0, max(p95s) * 1.30)
    ax.legend(loc="upper right", fontsize=9)

    p = FIG_DIR / "latency_comparison.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _axis_breakdown_bar() -> Path:
    """Per-session per-axis error to show distinct failure modes."""
    # Reload using the same script as outlier_diagnose; here we pull from a
    # cached JSON by re-running quickly via the saved per-session file is not
    # enough (it doesn't have per-axis) — recompute lightweight from scratch.
    import sys
    sys.path.insert(0, str(ROOT))
    import torch
    from torch.utils.data import DataLoader
    from depthpose.data.training_dataset import JOINT_ORDER, TrainingSession  # noqa
    from depthpose.model.student import DepthPoseStudent
    from depthpose.training.config import Config
    from collections import defaultdict

    cfg = Config.from_yaml(ROOT / "runs" / "run1_baseline" / "config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TrainingSession(
        raw_dir=cfg.data.raw_dir, labels_root=cfg.data.labels_dir,
        image_size=cfg.data.image_size, split_file=cfg.data.splits_path,
        split="test", drop_swaps=True,
    )
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.num_workers, pin_memory=True)
    model = DepthPoseStudent(
        backbone_name=cfg.model.backbone, num_joints=cfg.model.num_joints,
        num_deconv=cfg.model.num_deconv, deconv_channels=cfg.model.deconv_channels,
        softargmax_beta=cfg.model.softargmax_beta, pretrained=False,
    ).to(device)
    state = torch.load(ROOT / "runs" / "run1_baseline" / "best.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state["model"]); model.eval()

    by_sess = defaultdict(lambda: {"dx": [], "dy": [], "dz": []})
    with torch.inference_mode():
        for batch in loader:
            depth = batch["depth"].to(device); intr = batch["intrinsics_input"].to(device)
            out = model(depth, intr)
            pred = out["coords_3d"].cpu().numpy()
            tgt = batch["target_3d"].numpy()
            valid = (batch["depth_valid"] & ~batch["needs_review"]).numpy()
            for i in range(pred.shape[0]):
                m = valid[i]
                if not m.any(): continue
                d = (pred[i] - tgt[i])[m] * 1000.0
                k = batch["session"][i]
                by_sess[k]["dx"].extend(np.abs(d[:, 0]).tolist())
                by_sess[k]["dy"].extend(np.abs(d[:, 1]).tolist())
                by_sess[k]["dz"].extend(np.abs(d[:, 2]).tolist())

    sessions = sorted(by_sess.keys(), key=lambda k: int(k))
    dx = [float(np.mean(by_sess[k]["dx"])) for k in sessions]
    dy = [float(np.mean(by_sess[k]["dy"])) for k in sessions]
    dz = [float(np.mean(by_sess[k]["dz"])) for k in sessions]

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x = np.arange(len(sessions))
    w = 0.27
    ax.bar(x - w, dx, w, label="|dx| (lateral)", color="#1f77b4", edgecolor="k", lw=0.4)
    ax.bar(x,     dy, w, label="|dy| (vertical)", color="#2ca02c", edgecolor="k", lw=0.4)
    ax.bar(x + w, dz, w, label="|dz| (depth)",   color="#d62728", edgecolor="k", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S01/{s}" for s in sessions], fontsize=8, rotation=0)
    ax.set_ylabel("mean |error| per axis (mm)")
    ax.set_title("Per-axis error breakdown — S01/3 is depth-failure, S01/5 is lateral-failure")
    ax.legend(fontsize=9)

    p = FIG_DIR / "axis_breakdown.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


@app.command()
def main(
    skip_axis: bool = typer.Option(False, "--skip-axis",
                                    help="Skip the axis-breakdown fig (slow — runs the model)"),
) -> None:
    out = []
    out.append(_per_session_mpjpe_bar())
    out.append(_latency_bar())
    if not skip_axis:
        out.append(_axis_breakdown_bar())
    typer.echo("\nblog figures:")
    for p in out:
        typer.echo(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    app()
