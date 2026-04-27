"""Run a trained student over every frame of every session in order,
derive gait metrics per session per side, write JSON.

This is the "deployment" view of the student: at inference time we
have no oracle, so we run on every frame and derive gait directly
from predictions. The oracle's parquet supplies frame ordering and
intrinsics; it does NOT mask the student's predictions.

Note (caveat): the student saw 80 % of each session in training, so
the gait timeseries may show optimistic frame-level accuracy versus a
truly held-out session. The Phase-3 H2 test is whether *aggregated*
gait parameters survive the depth-only ground-truth gap, not a fresh
generalisation test. Document accordingly in the blog.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from depthpose.data.training_dataset import JOINT_ORDER, TrainingSession
from depthpose.eval.gait import derive_gait_metrics
from depthpose.model.student import DepthPoseStudent
from depthpose.training.config import Config

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


@torch.inference_mode()
def predict_session(
    model: torch.nn.Module,
    raw_dir: Path,
    labels_root: Path,
    image_size: tuple[int, int],
    subject: str,
    session: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (coords (T, J, 3), oracle_valid (T, J), frame_index (T,))."""
    ds = TrainingSession(
        raw_dir=raw_dir,
        labels_root=labels_root,
        image_size=image_size,
        sessions=[(subject, session)],
        split_file=None,
        split="all",
        drop_swaps=False,                 # we want every frame, in order
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    coords_list: list[np.ndarray] = []
    valid_list: list[np.ndarray] = []
    fi_list: list[int] = []
    model.eval()
    for batch in loader:
        depth = batch["depth"].to(device, non_blocking=True)
        intr = batch["intrinsics_input"].to(device, non_blocking=True)
        out = model(depth, intr)
        coords_list.append(out["coords_3d"].cpu().numpy())
        oracle_valid = (batch["depth_valid"] & ~batch["needs_review"]).numpy()
        valid_list.append(oracle_valid)
        fi_list.extend(int(x) for x in batch["frame_index"].tolist())
    coords = np.concatenate(coords_list, axis=0)
    valid = np.concatenate(valid_list, axis=0)
    order = np.argsort(fi_list)            # restore frame_index ordering
    return coords[order], valid[order], np.array(fi_list)[order]


@app.command()
def main(
    run_dir: Path = typer.Option(..., "--run-dir"),
    checkpoint: str = typer.Option("best", "--checkpoint"),
    out: Path = typer.Option(Path("reports/student_gait.json"), "--out"),
    fps: float = typer.Option(30.0, "--fps"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                        format="%(levelname)s %(message)s")
    cfg = Config.from_yaml(run_dir / "config.yaml")
    assert cfg.model is not None and cfg.training is not None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # discover sessions
    sessions = sorted(
        (p.parent.parent.name, p.parent.name)
        for p in cfg.data.raw_dir.glob("*/*/meta.json")
    )

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
    logger.info("loaded %s/%s.pt (epoch %d)", run_dir, checkpoint,
                state.get("epoch", -1))

    rows: list[dict] = []
    for subject, session in sessions:
        coords, oracle_valid, _ = predict_session(
            model=model,
            raw_dir=cfg.data.raw_dir,
            labels_root=cfg.data.labels_dir,
            image_size=cfg.data.image_size,
            subject=subject,
            session=session,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            device=device,
        )
        T, J = coords.shape[0], coords.shape[1]
        # The student is always "valid" since it predicts a Z everywhere.
        student_valid = np.ones((T, J), dtype=bool)
        for side in ("left", "right"):
            g = derive_gait_metrics(coords, student_valid, list(JOINT_ORDER),
                                     fps=fps, side=side)
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
            logger.info("%s/%s [%s]: %s", subject, session, side, entry["status"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    typer.echo(f"\nwrote {out} with {len(rows)} session-side rows")


if __name__ == "__main__":
    app()
