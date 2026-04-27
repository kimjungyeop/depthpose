"""Bootstrap confidence interval for held-out per-frame MPJPE.

Loads the run4 holdout model, runs inference on all 401 S01/14 frames,
computes per-frame MPJPE (averaged over the joints whose oracle label is
flagged ``depth_valid & ~needs_review``), then bootstraps over frames to
produce a 95% confidence interval.

Saves per-frame errors and CI bounds to
``reports/holdout_s01_14_bootstrap.json`` so the report's CI claim has
verifiable provenance.

Usage::

    python -m depthpose.eval.bootstrap_ci

Defaults: 5,000 bootstrap replicates, frame-level resampling, percentile
method (2.5/97.5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from depthpose.data.training_dataset import JOINT_ORDER, TrainingSession
from depthpose.eval.metrics import mpjpe_overall_mm
from depthpose.model.student import DepthPoseStudent
from depthpose.training.config import Config

ROOT = Path(__file__).resolve().parents[2]
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    run_dir: Path = typer.Option(Path("runs/run4_holdout_s01_14"), "--run-dir"),
    checkpoint: str = typer.Option("best", "--checkpoint"),
    subject: str = typer.Option("S01", "--subject"),
    session: str = typer.Option("14", "--session"),
    n_replicates: int = typer.Option(5000, "--n-replicates"),
    seed: int = typer.Option(0, "--seed"),
    out_json: Path = typer.Option(
        Path("reports/holdout_s01_14_bootstrap.json"), "--out-json"
    ),
) -> None:
    cfg = Config.from_yaml(run_dir / "config.yaml")
    assert cfg.model is not None and cfg.data is not None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthPoseStudent(
        backbone_name=cfg.model.backbone,
        num_joints=cfg.model.num_joints,
        num_deconv=cfg.model.num_deconv,
        deconv_channels=cfg.model.deconv_channels,
        softargmax_beta=cfg.model.softargmax_beta,
        pretrained=False,
    ).to(device)
    state = torch.load(run_dir / f"{checkpoint}.pt", map_location=device,
                       weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    ds = TrainingSession(
        raw_dir=cfg.data.raw_dir,
        labels_root=cfg.data.labels_dir,
        image_size=cfg.data.image_size,
        sessions=[(subject, session)],
        split_file=None,
        split="all",
        drop_swaps=False,
    )

    # Mirror the exact eval pipeline used by holdout_compare.py:
    # DataLoader with batch_size=64, then concatenate predictions and
    # targets, then compute MPJPE via metrics.mpjpe_overall_mm. This
    # guarantees the bootstrap mean matches the previously-reported 39.55
    # mm headline number.
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                        pin_memory=False)
    preds_l, tgts_l, valids_l, frame_idxs = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            out_b = model(batch["depth"].to(device),
                          batch["intrinsics_input"].to(device))
            preds_l.append(out_b["coords_3d"].cpu().numpy())
            tgts_l.append(batch["target_3d"].numpy())
            valids_l.append((batch["depth_valid"] & ~batch["needs_review"]).numpy())
            frame_idxs.extend(int(x) for x in batch["frame_index"].tolist())
    pred = np.concatenate(preds_l)            # (N, J, 3)  metres
    tgt = np.concatenate(tgts_l)              # (N, J, 3)  metres
    valid = np.concatenate(valids_l).astype(bool)  # (N, J)
    order = np.argsort(frame_idxs)
    pred = pred[order]; tgt = tgt[order]; valid = valid[order]

    # Headline (whole-set) joint-weighted MPJPE — should match 39.55 mm.
    mean_mpjpe = float(mpjpe_overall_mm(pred, tgt, valid))

    # Per-frame numerator (sum of valid-joint errors in mm) and count
    # used for the bootstrap.
    err_mm = np.linalg.norm(pred - tgt, axis=2) * 1000.0   # (N, J)
    err_sum = (err_mm * valid).sum(axis=1)                  # (N,) sum mm per frame
    n_valid_per_frame = valid.sum(axis=1)                   # (N,) ints

    # Drop any frames with zero valid joints (none expected here, but be safe).
    keep = n_valid_per_frame > 0
    err_sum = err_sum[keep].astype(np.float64)
    n_valid_per_frame = n_valid_per_frame[keep].astype(np.int64)
    per_frame_index = [int(frame_idxs[order[i]]) for i, k in enumerate(keep) if k]
    per_frame_mpjpe_mm = (err_sum / n_valid_per_frame).tolist()
    n_frames = int(keep.sum())
    n_valid_joints_total = int(n_valid_per_frame.sum())

    rng = np.random.default_rng(seed)
    boots = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        idx = rng.integers(0, n_frames, size=n_frames)
        boots[r] = float(err_sum[idx].sum() / n_valid_per_frame[idx].sum())
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))

    out = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "subject": subject,
        "session": session,
        "n_frames": n_frames,
        "n_valid_joints_total": n_valid_joints_total,
        "n_replicates": int(n_replicates),
        "seed": int(seed),
        "method": (
            "frame-level resample with replacement; "
            "joint-weighted MPJPE per resample = sum(err)/sum(n_valid); "
            "percentile 2.5/97.5"
        ),
        "mean_mpjpe_mm": mean_mpjpe,
        "ci95_lo_mm": ci_lo,
        "ci95_hi_mm": ci_hi,
        "per_frame_mpjpe_mm": [float(x) for x in per_frame_mpjpe_mm],
        "per_frame_n_valid_joints": [int(x) for x in n_valid_per_frame],
        "per_frame_index": per_frame_index,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    typer.echo(
        f"n_frames={n_frames}  mean={mean_mpjpe:.3f}  "
        f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  "
        f"({n_replicates} replicates)"
    )
    typer.echo(f"saved {out_json}")


if __name__ == "__main__":
    app()
