"""H2 hypothesis test: depth-only student gait params vs oracle gait params.

Loads ``reports/oracle_gait.json`` and ``reports/student_gait.json``,
joins on (subject, session, side), computes signed and absolute relative
errors per parameter, writes:

- ``reports/h2_compare.json``: machine-readable.
- ``reports/h2_compare.md``: markdown summary.
- ``reports/h2_scatter.png``: oracle (x) vs student (y) scatter for each
  gait parameter.
- ``reports/h2_bland_altman.png``: Bland-Altman style plot — diff vs mean.

H2 target (per the brief): <10 % relative error on cadence and stride.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


# Comparable parameters and a friendly label
COMPARED_PARAMS: list[tuple[str, str]] = [
    ("cadence_steps_per_min", "Cadence (steps/min)"),
    ("stride_period_s", "Stride period (s)"),
    ("apparent_step_amplitude_m", "Step amplitude (m)"),
    ("knee_flexion_max_deg", "Knee flex max (°)"),
    ("knee_flexion_range_deg", "Knee flex range (°)"),
]


def _key(r: dict) -> tuple[str, str, str]:
    return (str(r["subject"]), str(r["session"]), str(r["side"]))


def _join(oracle: list[dict], student: list[dict]) -> list[dict]:
    o_by_key = {_key(r): r for r in oracle}
    rows: list[dict] = []
    for s in student:
        k = _key(s)
        o = o_by_key.get(k)
        if o is None:
            continue
        if s.get("status") != "ok" or o.get("status") != "ok":
            continue
        merged: dict = {"subject": k[0], "session": k[1], "side": k[2]}
        for param, _ in COMPARED_PARAMS:
            merged[f"oracle_{param}"] = float(o[param])
            merged[f"student_{param}"] = float(s[param])
        rows.append(merged)
    return rows


def _abs_rel_err(o: float, s: float) -> float:
    if not np.isfinite(o) or not np.isfinite(s) or o == 0.0:
        return float("nan")
    return abs(s - o) / abs(o) * 100.0


@app.command()
def main(
    oracle_path: Path = typer.Option(Path("reports/oracle_gait.json"), "--oracle"),
    student_path: Path = typer.Option(Path("reports/student_gait.json"), "--student"),
    out_dir: Path = typer.Option(Path("reports"), "--out-dir"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                        format="%(levelname)s %(message)s")
    oracle = json.loads(oracle_path.read_text())
    student = json.loads(student_path.read_text())
    rows = _join(oracle, student)
    if not rows:
        raise typer.BadParameter("no joinable session-sides between oracle/student")
    logger.info("matched %d session-side pairs", len(rows))

    # ---- per-parameter aggregate ----
    summary: dict[str, dict] = {}
    for param, label in COMPARED_PARAMS:
        o = np.array([r[f"oracle_{param}"] for r in rows], dtype=float)
        s = np.array([r[f"student_{param}"] for r in rows], dtype=float)
        finite = np.isfinite(o) & np.isfinite(s)
        o = o[finite]; s = s[finite]
        if o.size == 0:
            continue
        signed_diff = s - o
        rel = np.array([_abs_rel_err(oo, ss) for oo, ss in zip(o, s)])
        rel = rel[np.isfinite(rel)]
        summary[param] = {
            "label": label,
            "n": int(o.size),
            "mae_units": float(np.mean(np.abs(signed_diff))),
            "rmse_units": float(np.sqrt(np.mean(signed_diff ** 2))),
            "median_signed_diff": float(np.median(signed_diff)),
            "abs_rel_err_pct_median": float(np.median(rel)) if rel.size else float("nan"),
            "abs_rel_err_pct_p90": float(np.percentile(rel, 90)) if rel.size else float("nan"),
            "abs_rel_err_pct_max": float(np.max(rel)) if rel.size else float("nan"),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "h2_compare.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )

    # ---- markdown ----
    H2_TARGETS = {"cadence_steps_per_min": 10, "stride_period_s": 10}
    lines: list[str] = ["# H2 — gait-parameter robustness", ""]
    lines.append(f"- session-sides matched: **{len(rows)}**")
    lines.append("- H2 target: <10% absolute relative error on cadence + stride")
    lines.append("")
    lines.append("| parameter | n | MAE | median rel err (%) | p90 rel err (%) | passes target? |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for param, label in COMPARED_PARAMS:
        s = summary.get(param)
        if s is None:
            continue
        target = H2_TARGETS.get(param)
        target_cell = (
            ("✓" if s["abs_rel_err_pct_median"] < target else "✗")
            if target is not None else "—"
        )
        lines.append(
            f"| {label} | {s['n']} | {s['mae_units']:.3f} | "
            f"{s['abs_rel_err_pct_median']:.1f} | "
            f"{s['abs_rel_err_pct_p90']:.1f} | {target_cell} |"
        )
    (out_dir / "h2_compare.md").write_text("\n".join(lines))

    # ---- scatter plot ----
    fig, axes = plt.subplots(1, len(COMPARED_PARAMS),
                              figsize=(3.4 * len(COMPARED_PARAMS), 3.4),
                              constrained_layout=True)
    for ax, (param, label) in zip(axes, COMPARED_PARAMS):
        o = np.array([r[f"oracle_{param}"] for r in rows], dtype=float)
        s = np.array([r[f"student_{param}"] for r in rows], dtype=float)
        finite = np.isfinite(o) & np.isfinite(s)
        ax.scatter(o[finite], s[finite], s=22, alpha=0.7,
                   color="#377eb8", edgecolor="white")
        lo = float(min(o[finite].min(), s[finite].min())) if finite.any() else 0.0
        hi = float(max(o[finite].max(), s[finite].max())) if finite.any() else 1.0
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel(f"oracle  {label}")
        ax.set_ylabel(f"student {label}")
        ax.set_title(label, fontsize=10)
    fig.suptitle("Oracle vs student — gait parameters (per session × side)", fontsize=11)
    fig.savefig(out_dir / "h2_scatter.png", dpi=120)
    plt.close(fig)

    # ---- Bland-Altman (diff vs mean) ----
    fig2, axes2 = plt.subplots(1, len(COMPARED_PARAMS),
                                figsize=(3.4 * len(COMPARED_PARAMS), 3.0),
                                constrained_layout=True)
    for ax, (param, label) in zip(axes2, COMPARED_PARAMS):
        o = np.array([r[f"oracle_{param}"] for r in rows], dtype=float)
        s = np.array([r[f"student_{param}"] for r in rows], dtype=float)
        finite = np.isfinite(o) & np.isfinite(s)
        m = (o[finite] + s[finite]) / 2
        d = s[finite] - o[finite]
        ax.scatter(m, d, s=22, alpha=0.7, color="#e41a1c", edgecolor="white")
        ax.axhline(0, color="black", lw=0.7)
        if d.size:
            ax.axhline(d.mean(), color="grey", lw=0.6, ls="--",
                       label=f"bias={d.mean():.2g}")
            ax.axhline(d.mean() + 1.96 * d.std(), color="grey", lw=0.5, ls=":")
            ax.axhline(d.mean() - 1.96 * d.std(), color="grey", lw=0.5, ls=":")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel(f"mean(o, s)  ({label})")
        ax.set_ylabel("student − oracle")
        ax.set_title(label, fontsize=10)
    fig2.suptitle("Bland-Altman — student vs oracle", fontsize=11)
    fig2.savefig(out_dir / "h2_bland_altman.png", dpi=120)
    plt.close(fig2)

    # ---- console summary ----
    typer.echo("\n=== H2 summary ===")
    for param, s in summary.items():
        typer.echo(
            f"  {s['label']:<26}  n={s['n']:>2}  MAE={s['mae_units']:.3f}  "
            f"rel-err median={s['abs_rel_err_pct_median']:.1f}%  "
            f"p90={s['abs_rel_err_pct_p90']:.1f}%"
        )


if __name__ == "__main__":
    app()
