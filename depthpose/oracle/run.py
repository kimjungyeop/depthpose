"""Run the ViTPose oracle over one or more sessions, write a parquet of
2D + 3D + confidence per joint.

Output schema (one row per (frame, joint)):

    subject              str
    session              str
    frame_index          int32
    joint_name           str
    u_px                 float32     (oracle 2D in image pixels)
    v_px                 float32
    conf_2d              float32     (ViTPose confidence in [0, 1])
    x_m, y_m, z_m        float32     (camera-frame 3D in metres; NaN if invalid)
    depth_valid          bool        (median-sampled depth was nonzero)
    needs_review         bool        (conf_2d < conf_threshold)

The parquet lives at ``<labels_root>/<subject>/<session>.parquet``.

Hip-depth policy (per project memory): when ``depth_valid`` is False
the 3D coords are NaN. Never extrapolate.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from depthpose.data.dataset import WalkerSession
from depthpose.oracle.lift import lift_keypoints_2d_to_3d
from depthpose.oracle.vitpose import ViTPoseOracle

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def run_oracle_on_session(
    raw_dir: Path,
    subject: str,
    session: str,
    out_path: Path,
    oracle: ViTPoseOracle,
    conf_threshold: float = 0.5,
    median_kernel: int = 3,
) -> pd.DataFrame:
    """Run the oracle over every frame of one session, return + write parquet."""
    ds = WalkerSession(raw_dir, sessions=[(subject, session)])
    n = len(ds)
    joint_names = oracle.joint_names
    joints_per_row = len(joint_names)
    n_rows = n * joints_per_row

    cols: dict[str, np.ndarray] = {
        "subject":      np.empty(n_rows, dtype=object),
        "session":      np.empty(n_rows, dtype=object),
        "frame_index":  np.empty(n_rows, dtype=np.int32),
        "joint_name":   np.empty(n_rows, dtype=object),
        "u_px":         np.empty(n_rows, dtype=np.float32),
        "v_px":         np.empty(n_rows, dtype=np.float32),
        "conf_2d":      np.empty(n_rows, dtype=np.float32),
        "x_m":          np.full(n_rows, np.nan, dtype=np.float32),
        "y_m":          np.full(n_rows, np.nan, dtype=np.float32),
        "z_m":          np.full(n_rows, np.nan, dtype=np.float32),
        "depth_valid":  np.zeros(n_rows, dtype=bool),
        "needs_review": np.zeros(n_rows, dtype=bool),
    }

    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        kp = oracle.detect(sample["rgb"])
        coords3d, valid = lift_keypoints_2d_to_3d(
            kp.coords,
            sample["depth_mm"],
            sample["intrinsics"],
            median_kernel=median_kernel,
        )
        for j, name in enumerate(joint_names):
            row = i * joints_per_row + j
            cols["subject"][row] = sample["subject"]
            cols["session"][row] = sample["session"]
            cols["frame_index"][row] = sample["frame_index"]
            cols["joint_name"][row] = name
            cols["u_px"][row] = float(kp.coords[j, 0])
            cols["v_px"][row] = float(kp.coords[j, 1])
            cols["conf_2d"][row] = float(kp.scores[j])
            cols["depth_valid"][row] = bool(valid[j])
            cols["needs_review"][row] = bool(kp.scores[j] < conf_threshold)
            if valid[j]:
                cols["x_m"][row] = float(coords3d[j, 0])
                cols["y_m"][row] = float(coords3d[j, 1])
                cols["z_m"][row] = float(coords3d[j, 2])

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            logger.info("  %d/%d frames (%.1f fps, eta %.0fs)", i + 1, n, rate, eta)

    df = pd.DataFrame(cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


@app.command()
def main(
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    labels_root: Path = typer.Option(Path("data/labels"), "--labels-root"),
    sessions: list[str] | None = typer.Option(
        None, "--session", "-s",
        help="Subject:session pair, e.g. 'S01:1'. Repeatable. Defaults to all.",
    ),
    checkpoint: str = typer.Option("usyd-community/vitpose-plus-base", "--checkpoint"),
    conf_threshold: float = typer.Option(0.5, "--conf-threshold", min=0.0, max=1.0),
    median_kernel: int = typer.Option(3, "--median-kernel"),
    device: str | None = typer.Option(None, "--device"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run the ViTPose oracle over every selected session."""
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                        format="%(levelname)s %(message)s")

    if sessions is None:
        ds = WalkerSession(raw_dir)
        keys = sorted({(it[1], it[2]) for it in ds._items})  # (subject, session)
    else:
        keys = []
        for s in sessions:
            if ":" not in s:
                raise typer.BadParameter(f"--session must be 'subject:session', got {s!r}")
            subj, sess = s.split(":", 1)
            keys.append((subj, sess))

    oracle = ViTPoseOracle(checkpoint=checkpoint, device=device)

    summary: list[dict] = []
    for subj, sess in keys:
        out = labels_root / subj / f"{sess}.parquet"
        logger.info("session %s/%s → %s", subj, sess, out)
        df = run_oracle_on_session(
            raw_dir=raw_dir,
            subject=subj,
            session=sess,
            out_path=out,
            oracle=oracle,
            conf_threshold=conf_threshold,
            median_kernel=median_kernel,
        )
        n_frames = df["frame_index"].nunique()
        n_joints_total = len(df)
        all_valid_mask = (
            df.assign(valid=df["depth_valid"] & (~df["needs_review"]))
            .groupby("frame_index")["valid"].all()
        )
        n_all_valid_frames = int(all_valid_mask.sum())
        summary.append({
            "subject": subj,
            "session": sess,
            "n_frames": n_frames,
            "n_joints_total": n_joints_total,
            "frac_depth_valid": float(df["depth_valid"].mean()),
            "frac_needs_review": float(df["needs_review"].mean()),
            "n_frames_all_valid": n_all_valid_frames,
            "frac_frames_all_valid": (n_all_valid_frames / n_frames) if n_frames else 0.0,
            "out_path": str(out),
        })
        logger.info(
            "  done: %d frames, depth_valid=%.3f, needs_review=%.3f, "
            "all-joints-valid frames=%d/%d (%.1f%%)",
            n_frames, summary[-1]["frac_depth_valid"], summary[-1]["frac_needs_review"],
            n_all_valid_frames, n_frames, 100 * summary[-1]["frac_frames_all_valid"],
        )

    typer.echo("\n=== oracle summary ===")
    for r in summary:
        typer.echo(
            f"{r['subject']}/{r['session']}: {r['n_frames']} frames, "
            f"depth_valid={r['frac_depth_valid']:.3f}, "
            f"needs_review={r['frac_needs_review']:.3f}, "
            f"all-valid frames {r['n_frames_all_valid']}/{r['n_frames']} "
            f"({100 * r['frac_frames_all_valid']:.1f}%)"
        )


if __name__ == "__main__":
    app()
