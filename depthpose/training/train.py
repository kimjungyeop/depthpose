"""Training loop for the depth-only student.

Reads a single ``configs/<name>.yaml``, builds train + val DataLoaders
from the same ``WalkerSession`` + parquet GT pipeline, optimises with
AdamW + cosine schedule (linear warmup), logs to TensorBoard, and
checkpoints best val MPJPE on a per-epoch basis.

Per the project memory: per-joint masking via ``depth_valid``, never
whole-frame filtering. The same masking is used for both training loss
and the validation MPJPE metric.

Usage::

    python -m depthpose.training.train --config configs/default.yaml
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from depthpose.data.training_dataset import TrainingSession, JOINT_ORDER
from depthpose.model.student import DepthPoseStudent
from depthpose.training.config import Config
from depthpose.training.loss import depth_pose_loss

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


# ----------------------- helpers ---------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def mpjpe_per_joint_mm(
    pred: torch.Tensor,        # (B, J, 3) metres
    target: torch.Tensor,      # (B, J, 3) metres
    valid: torch.Tensor,       # (B, J) bool
) -> dict[str, float]:
    """Per-joint MPJPE in millimetres, masking by ``valid``.

    Returns a dict with ``mpjpe_mm_overall`` (mean over all valid joints
    across all joints) and ``mpjpe_mm_<joint_name>`` per joint.
    """
    err = torch.linalg.norm(pred - target, dim=-1)  # (B, J)
    out: dict[str, float] = {}
    overall_sum = 0.0
    overall_n = 0
    for j, name in enumerate(JOINT_ORDER):
        m = valid[:, j]
        if m.any():
            mean_m = err[:, j][m].mean().item()
            mm = mean_m * 1000.0
            out[f"mpjpe_mm_{name}"] = mm
            overall_sum += err[:, j][m].sum().item() * 1000.0
            overall_n += int(m.sum().item())
    out["mpjpe_mm_overall"] = overall_sum / max(overall_n, 1)
    out["n_valid"] = float(overall_n)
    return out


# --------------------- evaluation ---------------------------


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    head_stride: int,
    aux_heatmap_loss_weight: float,
    anatomical_loss_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    err_sums = torch.zeros(len(JOINT_ORDER), device=device)
    valid_counts = torch.zeros(len(JOINT_ORDER), device=device)
    loss_sum = 0.0
    n_batches = 0
    n_frames = 0
    n_crossover = 0
    for batch in loader:
        depth = batch["depth"].to(device, non_blocking=True)
        intr = batch["intrinsics_input"].to(device, non_blocking=True)
        target_3d = batch["target_3d"].to(device, non_blocking=True)
        target_uv = batch["target_uv_input"].to(device, non_blocking=True)
        depth_valid = batch["depth_valid"].to(device, non_blocking=True)
        needs_review = batch["needs_review"].to(device, non_blocking=True)

        pred = model(depth, intr)
        losses = depth_pose_loss(
            pred, target_3d, target_uv, depth_valid, needs_review,
            head_stride=head_stride,
            aux_heatmap_loss_weight=aux_heatmap_loss_weight,
            anatomical_loss_weight=anatomical_loss_weight,
        )
        loss_sum += float(losses["total"].item())
        n_batches += 1

        err = torch.linalg.norm(pred["coords_3d"] - target_3d, dim=-1)  # (B, J)
        valid = depth_valid & ~needs_review
        err_sums += (err * valid).sum(dim=0)
        valid_counts += valid.sum(dim=0)

        # Crossover monitoring: hip lateral order vs knee lateral order.
        x = pred["coords_3d"][..., 0]
        dx_hip  = x[..., 0] - x[..., 1]   # left_hip - right_hip
        dx_knee = x[..., 2] - x[..., 3]
        n_frames += int(x.shape[0])
        n_crossover += int(((dx_hip * dx_knee) < 0).sum().item())

    out: dict[str, float] = {
        "val_loss": loss_sum / max(n_batches, 1),
        "frac_hip_knee_crossover": n_crossover / max(n_frames, 1),
    }
    overall_sum = 0.0
    overall_n = 0
    for j, name in enumerate(JOINT_ORDER):
        if valid_counts[j] > 0:
            mm = (err_sums[j] / valid_counts[j]).item() * 1000.0
            out[f"mpjpe_mm_{name}"] = mm
            overall_sum += err_sums[j].item() * 1000.0
            overall_n += int(valid_counts[j].item())
    out["mpjpe_mm_overall"] = overall_sum / max(overall_n, 1)
    out["n_valid"] = float(overall_n)
    return out


# ----------------------- trainer ----------------------------


def train(
    config_path: Path,
    run_name: str | None = None,
    epochs_override: int | None = None,
) -> Path:
    cfg = Config.from_yaml(config_path)
    assert cfg.model is not None and cfg.training is not None
    set_global_seed(cfg.project.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if run_name is None:
        run_name = time.strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.training.log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    # --- data ---
    train_ds = TrainingSession(
        raw_dir=cfg.data.raw_dir,
        labels_root=cfg.data.labels_dir,
        image_size=cfg.data.image_size,
        split_file=cfg.data.splits_path,
        split="train",
        drop_swaps=True,
    )
    val_ds = TrainingSession(
        raw_dir=cfg.data.raw_dir,
        labels_root=cfg.data.labels_dir,
        image_size=cfg.data.image_size,
        split_file=cfg.data.splits_path,
        split="test",
        drop_swaps=True,
    )
    logger.info("dataset: train=%d, val=%d", len(train_ds), len(val_ds))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # --- model ---
    model = DepthPoseStudent(
        backbone_name=cfg.model.backbone,
        num_joints=cfg.model.num_joints,
        num_deconv=cfg.model.num_deconv,
        deconv_channels=cfg.model.deconv_channels,
        softargmax_beta=cfg.model.softargmax_beta,
        pretrained=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model: %.2f M params, head_stride=%d", n_params / 1e6, model.head_stride)

    # --- optim / sched ---
    epochs = epochs_override or cfg.training.num_epochs
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * cfg.training.warmup_pct))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # --- train ---
    best_mpjpe_mm = float("inf")
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history: list[dict[str, Any]] = []

    global_step = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_l3d = 0.0
        epoch_n = 0
        for batch in train_loader:
            lr = cosine_lr(global_step, total_steps, warmup_steps, cfg.training.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            depth = batch["depth"].to(device, non_blocking=True)
            intr = batch["intrinsics_input"].to(device, non_blocking=True)
            target_3d = batch["target_3d"].to(device, non_blocking=True)
            target_uv = batch["target_uv_input"].to(device, non_blocking=True)
            depth_valid = batch["depth_valid"].to(device, non_blocking=True)
            needs_review = batch["needs_review"].to(device, non_blocking=True)

            pred = model(depth, intr)
            losses = depth_pose_loss(
                pred, target_3d, target_uv, depth_valid, needs_review,
                head_stride=model.head_stride,
                aux_heatmap_loss_weight=cfg.training.aux_heatmap_loss_weight,
                anatomical_loss_weight=cfg.training.anatomical_loss_weight,
            )
            loss = losses["total"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.training.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                cfg.training.grad_clip_norm)
            opt.step()

            epoch_loss += float(loss.item())
            epoch_l3d += float(losses["l3d"].item())
            epoch_n += 1
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/l3d", float(losses["l3d"].item()), global_step)
            if "l_anat" in losses:
                writer.add_scalar("train/l_anat", float(losses["l_anat"].item()),
                                  global_step)
            writer.add_scalar("train/lr", lr, global_step)
            global_step += 1

        train_loss = epoch_loss / max(epoch_n, 1)
        train_l3d = epoch_l3d / max(epoch_n, 1)
        val = evaluate(
            model, val_loader, device, model.head_stride,
            cfg.training.aux_heatmap_loss_weight,
            cfg.training.anatomical_loss_weight,
        )
        writer.add_scalar("val/loss", val["val_loss"], global_step)
        writer.add_scalar("val/mpjpe_mm_overall", val["mpjpe_mm_overall"], global_step)
        writer.add_scalar("val/frac_hip_knee_crossover",
                          val["frac_hip_knee_crossover"], global_step)
        for j in JOINT_ORDER:
            k = f"mpjpe_mm_{j}"
            if k in val:
                writer.add_scalar(f"val/{k}", val[k], global_step)

        elapsed = time.time() - t0
        eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
        logger.info(
            "ep %3d/%d  train_loss=%.4f  l3d=%.4f  val_loss=%.4f  "
            "MPJPE=%.1f mm (best=%.1f)  cross=%.2f%%  lr=%.2e  elapsed=%.0fs eta=%.0fs",
            epoch + 1, epochs, train_loss, train_l3d, val["val_loss"],
            val["mpjpe_mm_overall"], min(best_mpjpe_mm, val["mpjpe_mm_overall"]),
            val["frac_hip_knee_crossover"] * 100, lr, elapsed, eta,
        )
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_l3d": train_l3d,
            "val_loss": val["val_loss"], "mpjpe_mm_overall": val["mpjpe_mm_overall"],
            "frac_hip_knee_crossover": val["frac_hip_knee_crossover"],
            **{k: v for k, v in val.items() if k.startswith("mpjpe_mm_")},
            "lr": lr,
        })

        # checkpointing
        torch.save({
            "epoch": epoch + 1, "model": model.state_dict(),
            "opt": opt.state_dict(), "val": val,
        }, last_path)
        if val["mpjpe_mm_overall"] < best_mpjpe_mm:
            best_mpjpe_mm = val["mpjpe_mm_overall"]
            torch.save({
                "epoch": epoch + 1, "model": model.state_dict(),
                "val": val,
            }, best_path)

    writer.close()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "run_dir": str(run_dir),
        "epochs": epochs,
        "best_mpjpe_mm": best_mpjpe_mm,
        "final_val": history[-1] if history else {},
        "n_params": n_params,
        "elapsed_s": time.time() - t0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("done — best MPJPE %.1f mm; run dir: %s", best_mpjpe_mm, run_dir)
    return run_dir


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    run_name: str | None = typer.Option(None, "--run-name"),
    epochs: int | None = typer.Option(None, "--epochs"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                        format="%(asctime)s %(levelname)s %(message)s")
    train(config_path=config, run_name=run_name, epochs_override=epochs)


if __name__ == "__main__":
    app()
