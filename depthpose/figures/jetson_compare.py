"""Student-vs-oracle latency comparison on Jetson Xavier NX (projected).

Three groupings on the x-axis:
  1. RTX A5000, FP32                — measured
  2. RTX A5000, FP16                — measured
  3. Jetson Xavier NX, TRT FP16     — projected (range, hatched bars)

For each grouping, two bars: oracle (ViTPose++ base, 125 M params) vs
student (depth-only MobileNetV2, 5.2 M params). Numbers come from:

  - runs/run3_anatomical/latency_benchmark.json  (student, RTX A5000)
  - reports/oracle_latency_benchmark.json        (oracle, RTX A5000)

Jetson NX projections come from the spec-ratio + published-benchmark
triangulation discussed in the blog. Student: 5–10 ms TRT FP16. Oracle:
50–100 ms TRT FP16. Plotted as range bars with the median point shown.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import typer

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main() -> None:
    student = json.loads(
        (ROOT / "runs/run3_anatomical/latency_benchmark.json").read_text()
    )
    oracle = json.loads(
        (ROOT / "reports/oracle_latency_benchmark.json").read_text()
    )

    # --- Workstation numbers (measured, milliseconds, batch 1, model-only where possible) ---
    student_a5000_fp32 = student["results_gpu"]["onnxruntime_cuda_fp32"]["median_ms"]   # 0.95
    student_a5000_fp16 = student["results_gpu"]["pytorch_cuda_fp16_autocast"]["median_ms"]  # ~4.0 (autocast overhead dominates on tiny model)

    oracle_a5000_fp32 = oracle["results"]["pytorch_cuda_fp32_model_only"]["median"]      # 7.93
    oracle_a5000_fp16 = oracle["results"]["pytorch_cuda_fp16_model_only"]["median"]      # 9.38

    # --- Jetson Xavier NX projections (range, ms) ---
    student_jetson_lo, student_jetson_hi = 5.0, 10.0
    oracle_jetson_lo, oracle_jetson_hi = 50.0, 100.0

    groups = [
        ("RTX A5000\nFP32 (measured)",       oracle_a5000_fp32, student_a5000_fp32, False),
        ("RTX A5000\nFP16 (measured)",       oracle_a5000_fp16, student_a5000_fp16, False),
        ("Jetson Xavier NX\nTRT FP16 (projected)",
            (oracle_jetson_lo + oracle_jetson_hi) / 2,
            (student_jetson_lo + student_jetson_hi) / 2,
            True),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    x = np.arange(len(groups))
    w = 0.35
    oracle_color = "#9467bd"   # ViT — purple
    student_color = "#2ca02c"  # depth — green

    for i, (label, ov, sv, hatched) in enumerate(groups):
        hatch = "//" if hatched else None
        ax.bar(x[i] - w / 2, ov, w, color=oracle_color, hatch=hatch,
               edgecolor="black", linewidth=0.6,
               label="Oracle (ViTPose++ base, 125 M params)" if i == 0 else None)
        ax.bar(x[i] + w / 2, sv, w, color=student_color, hatch=hatch,
               edgecolor="black", linewidth=0.6,
               label="Student (depth-only, 5.2 M params)" if i == 0 else None)

    # Range whiskers on the projected Jetson bars.
    jetson_x = x[2]
    ax.errorbar([jetson_x - w / 2], [(oracle_jetson_lo + oracle_jetson_hi) / 2],
                yerr=[[((oracle_jetson_hi - oracle_jetson_lo) / 2)],
                      [((oracle_jetson_hi - oracle_jetson_lo) / 2)]],
                fmt="none", capsize=6, color="black", lw=1.2)
    ax.errorbar([jetson_x + w / 2], [(student_jetson_lo + student_jetson_hi) / 2],
                yerr=[[((student_jetson_hi - student_jetson_lo) / 2)],
                      [((student_jetson_hi - student_jetson_lo) / 2)]],
                fmt="none", capsize=6, color="black", lw=1.2)

    # Annotate every bar with its value.
    for i, (label, ov, sv, hatched) in enumerate(groups):
        if hatched:
            o_label = f"{oracle_jetson_lo:.0f}–{oracle_jetson_hi:.0f} ms"
            s_label = f"{student_jetson_lo:.0f}–{student_jetson_hi:.0f} ms"
        else:
            o_label = f"{ov:.1f} ms"
            s_label = f"{sv:.1f} ms"
        ax.text(x[i] - w / 2, ov + 2, o_label, ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + w / 2, sv + 2, s_label, ha="center", va="bottom", fontsize=9)

    # 30 fps real-time line.
    ax.axhline(33.3, color="red", lw=1.2, ls="--", alpha=0.7,
               label="30 fps real-time budget (33.3 ms)")

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylabel("inference latency, batch 1 (ms — lower is better)")
    ax.set_title(
        "Student is roughly 10× faster than the oracle on the same edge silicon\n"
        "Hatched bars are projected from spec-ratio scaling + published Jetson benchmarks"
    )
    ax.set_ylim(0, max(oracle_jetson_hi, 33.3) * 1.20)

    # Build legend from existing handles + a hatched-patch entry for projection.
    handles, labels = ax.get_legend_handles_labels()
    proj_patch = mpatches.Patch(facecolor="#dddddd", hatch="//",
                                 edgecolor="black", label="projected (range)")
    handles.append(proj_patch); labels.append("projected (range)")
    ax.legend(handles, labels, fontsize=9, loc="upper left")

    out = FIG_DIR / "jetson_oracle_vs_student.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    app()
