"""Run gait derivations against the oracle parquets, per session.

Phase-3 baseline: gives the "ground-truth" gait parameters that the
student's gait outputs will be compared against. Uses ``derive_gait_metrics``
on each session's oracle (J, 3) sequence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from depthpose.data.training_dataset import JOINT_ORDER
from depthpose.eval.gait import derive_gait_metrics

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def session_oracle_sequence(
    parquet_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords (T, J, 3), valid (T, J)) for one session, sorted by frame_index."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values(["frame_index", "joint_name"])
    frames = sorted(df["frame_index"].unique())
    T = len(frames)
    J = len(JOINT_ORDER)
    coords = np.full((T, J, 3), np.nan, dtype=np.float32)
    valid = np.zeros((T, J), dtype=bool)
    name_to_j = {n: i for i, n in enumerate(JOINT_ORDER)}
    by_frame = df.groupby("frame_index")
    for ti, fi in enumerate(frames):
        g = by_frame.get_group(fi)
        for _, row in g.iterrows():
            jname = str(row["joint_name"])
            if jname not in name_to_j:
                continue  # ignore extra wholebody joints if any
            j = name_to_j[jname]
            valid[ti, j] = bool(row["depth_valid"]) and not bool(row["needs_review"])
            if valid[ti, j]:
                coords[ti, j] = (
                    float(row["x_m"]), float(row["y_m"]), float(row["z_m"])
                )
    return coords, valid


@app.command()
def main(
    labels_root: Path = typer.Option(Path("data/labels"), "--labels-root"),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    out: Path = typer.Option(Path("reports/oracle_gait.json"), "--out"),
    fps: float = typer.Option(30.0, "--fps"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                        format="%(levelname)s %(message)s")

    rows: list[dict] = []
    parts = sorted(labels_root.glob("*/*.parquet"))
    for p in parts:
        df = pd.read_parquet(p)
        subject = str(df.subject.iloc[0])
        session = str(df.session.iloc[0])
        coords, valid = session_oracle_sequence(p)

        for side in ("left", "right"):
            g = derive_gait_metrics(coords, valid, list(JOINT_ORDER), fps=fps, side=side)
            entry = {"subject": subject, "session": session, "side": side}
            if g is None:
                entry["status"] = "no_peaks"
            else:
                entry.update({
                    "status": "ok",
                    "n_steps": g.n_steps,
                    "duration_s": g.duration_s,
                    "cadence_steps_per_min": g.cadence_steps_per_min,
                    "stride_period_s": g.stride_period_s,
                    "apparent_step_amplitude_m": g.apparent_step_amplitude_m,
                    "knee_flexion_min_deg": g.knee_flexion_min_deg,
                    "knee_flexion_max_deg": g.knee_flexion_max_deg,
                    "knee_flexion_range_deg": g.knee_flexion_range_deg,
                })
            rows.append(entry)
            logger.info("%s/%s [%s]: %s", subject, session, side, entry.get("status"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    typer.echo(f"\nwrote {out} with {len(rows)} entries (14 sessions × 2 sides)")
    # quick summary
    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        cad = np.array([r["cadence_steps_per_min"] for r in ok])
        knee = np.array([r["knee_flexion_range_deg"] for r in ok if not np.isnan(r["knee_flexion_range_deg"])])
        amp = np.array([r["apparent_step_amplitude_m"] for r in ok])
        typer.echo(f"\noracle gait summary across {len(ok)} session-sides:")
        typer.echo(f"  cadence: median {np.median(cad):.1f}, range [{cad.min():.1f}, {cad.max():.1f}] steps/min")
        if len(knee):
            typer.echo(f"  knee flex range: median {np.median(knee):.1f}°, "
                       f"range [{knee.min():.1f}°, {knee.max():.1f}°]")
        typer.echo(f"  apparent step amplitude: median {np.median(amp):.3f} m")


if __name__ == "__main__":
    app()
