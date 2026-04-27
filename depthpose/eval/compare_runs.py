"""Compare evaluation outputs from two runs side-by-side.

Reads ``eval.json`` from each run, prints a delta table, and writes a
combined PCK-curve PNG with both runs overlaid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    run_a: Path = typer.Option(..., "--run-a"),
    run_b: Path = typer.Option(..., "--run-b"),
    out: Path | None = typer.Option(None, "--out", help="Path for the combined PCK PNG. "
                                                          "Defaults to next to run_b."),
) -> None:
    a = json.loads((run_a / "eval.json").read_text())
    b = json.loads((run_b / "eval.json").read_text())
    name_a = run_a.name
    name_b = run_b.name

    print(f"\n=== {name_a}  vs  {name_b} ===")
    print(f"  MPJPE overall: {a['mpjpe_mm_overall']:.1f} → {b['mpjpe_mm_overall']:.1f} mm "
          f"(Δ {b['mpjpe_mm_overall']-a['mpjpe_mm_overall']:+.1f})")
    print()
    print(f"{'joint':<14}  {name_a:>10}  {name_b:>10}  {'Δ (mm)':>10}")
    for j, va in a["mpjpe_mm_per_joint"].items():
        vb = b["mpjpe_mm_per_joint"][j]
        print(f"  {j:<14} {va:>10.1f}  {vb:>10.1f}  {vb-va:>+10.1f}")
    print()
    for t_label, va_pck in a["pck_overall"].items():
        vb_pck = b["pck_overall"][t_label]
        print(f"  PCK@{t_label:<5} overall  {va_pck*100:>6.1f}%  "
              f"{vb_pck*100:>6.1f}%   Δ {(vb_pck-va_pck)*100:+.1f}pp")

    if out is None:
        out = run_b / f"compare_{name_a}_vs_{name_b}.png"
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ts_a = np.array(a["pck_curve"]["thresholds_mm"])
    pck_a = np.array(a["pck_curve"]["pck"])
    ts_b = np.array(b["pck_curve"]["thresholds_mm"])
    pck_b = np.array(b["pck_curve"]["pck"])
    ax.plot(ts_a, pck_a, lw=2, label=f"{name_a} (MPJPE {a['mpjpe_mm_overall']:.1f} mm)")
    ax.plot(ts_b, pck_b, lw=2, label=f"{name_b} (MPJPE {b['mpjpe_mm_overall']:.1f} mm)")
    for t in (5, 10, 20, 50):
        ax.axvline(t, color="grey", lw=0.5, alpha=0.4)
    ax.set_xlabel("threshold (mm)")
    ax.set_ylabel("PCK")
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.0)
    ax.set_title(f"PCK comparison")
    ax.legend(loc="lower right")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    app()
