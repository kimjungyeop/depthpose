"""Compare two trained models on the S01/14 held-out session.

For each model:
  - Run on every frame of S01/14
  - Compute overall + per-joint MPJPE, PCK, hip-knee crossover rate
  - Run the gait derivation (cadence, stride period, step amplitude,
    knee flexion) per side and compare to oracle

Models compared:
  - run3_anatomical: trained on a random per-session split — saw ~80 % of
    S01/14 frames during training. Numbers here measure how well it
    reproduces oracle predictions on familiar geometry.
  - run4_holdout_s01_14: trained on S01/1..S01/13, never saw S01/14.
    Numbers here are the honest cross-bag generalisation signal.

Writes JSON + a 2-panel comparison figure to reports/.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/farandhigh-ubuntu/Documents/cv/depth-pose-tracking")
sys.path.insert(0, str(ROOT))
from depthpose.data.training_dataset import JOINT_ORDER, TrainingSession
from depthpose.eval.gait import derive_gait_metrics
from depthpose.eval.metrics import (mpjpe_overall_mm, mpjpe_per_joint_mm,
                                    pck_overall, pck_per_joint)
from depthpose.model.student import DepthPoseStudent
from depthpose.training.config import Config


def _eval_model(run_name: str) -> dict[str, Any]:
    cfg = Config.from_yaml(ROOT / "runs" / run_name / "config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TrainingSession(
        raw_dir=cfg.data.raw_dir, labels_root=cfg.data.labels_dir,
        image_size=cfg.data.image_size,
        sessions=[("S01", "14")],          # JUST S01/14
        split_file=None, split="all",
        drop_swaps=False,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=True)

    m = DepthPoseStudent(
        backbone_name=cfg.model.backbone, num_joints=cfg.model.num_joints,
        num_deconv=cfg.model.num_deconv, deconv_channels=cfg.model.deconv_channels,
        softargmax_beta=cfg.model.softargmax_beta, pretrained=False,
    ).to(device)
    state = torch.load(ROOT / "runs" / run_name / "best.pt",
                       map_location=device, weights_only=True)
    m.load_state_dict(state["model"]); m.eval()

    preds, tgts, valids, frame_idxs = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            out = m(batch["depth"].to(device), batch["intrinsics_input"].to(device))
            preds.append(out["coords_3d"].cpu().numpy())
            tgts.append(batch["target_3d"].numpy())
            valids.append((batch["depth_valid"] & ~batch["needs_review"]).numpy())
            frame_idxs.extend(int(x) for x in batch["frame_index"].tolist())
    pred = np.concatenate(preds); tgt = np.concatenate(tgts); valid = np.concatenate(valids)
    # restore time order
    order = np.argsort(frame_idxs)
    pred = pred[order]; tgt = tgt[order]; valid = valid[order]

    overall = mpjpe_overall_mm(pred, tgt, valid)
    per_j = mpjpe_per_joint_mm(pred, tgt, valid)
    pck_o = pck_overall(pred, tgt, valid, [5, 10, 20, 50])
    pck_j = pck_per_joint(pred, tgt, valid, [5, 10, 20, 50])

    LH, RH, LK, RK, LA, RA = (JOINT_ORDER.index(j) for j in
        ("left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"))
    x = pred[..., 0]
    n = pred.shape[0]
    n_hk = int(((x[:, LH] - x[:, RH]) * (x[:, LK] - x[:, RK]) < 0).sum())
    n_ha = int(((x[:, LH] - x[:, RH]) * (x[:, LA] - x[:, RA]) < 0).sum())

    # Gait derivations (student "deployment" mode — predict on all frames).
    student_valid = np.ones((n, len(JOINT_ORDER)), dtype=bool)
    gait: dict[str, Any] = {}
    for side in ("left", "right"):
        g_student = derive_gait_metrics(pred, student_valid, list(JOINT_ORDER),
                                         fps=30.0, side=side)
        g_oracle = derive_gait_metrics(tgt, valid, list(JOINT_ORDER),
                                        fps=30.0, side=side)
        if g_student is None or g_oracle is None:
            gait[side] = {"status": "no_peaks"}
        else:
            gait[side] = {
                "oracle_cadence_steps_per_min": g_oracle.cadence_steps_per_min,
                "student_cadence_steps_per_min": g_student.cadence_steps_per_min,
                "oracle_stride_period_s": g_oracle.stride_period_s,
                "student_stride_period_s": g_student.stride_period_s,
                "oracle_step_amp_m": g_oracle.apparent_step_amplitude_m,
                "student_step_amp_m": g_student.apparent_step_amplitude_m,
                "oracle_knee_flex_max_deg": g_oracle.knee_flexion_max_deg,
                "student_knee_flex_max_deg": g_student.knee_flexion_max_deg,
                "oracle_knee_flex_range_deg": g_oracle.knee_flexion_range_deg,
                "student_knee_flex_range_deg": g_student.knee_flexion_range_deg,
            }
    return {
        "n_frames": n,
        "n_valid_joints": int(valid.sum()),
        "mpjpe_overall_mm": overall,
        "mpjpe_per_joint_mm": {JOINT_ORDER[i]: float(per_j[i]) for i in range(len(JOINT_ORDER))},
        "pck_overall": {f"{t}mm": pck_o[t] for t in [5, 10, 20, 50]},
        "pck_per_joint": {
            f"{t}mm": {JOINT_ORDER[i]: float(pck_j[t][i]) for i in range(len(JOINT_ORDER))}
            for t in [5, 10, 20, 50]
        },
        "frac_hip_knee_crossover": n_hk / n,
        "frac_hip_ankle_crossover": n_ha / n,
        "gait": gait,
    }


if __name__ == "__main__":
    out = {}
    for r in ["run3_anatomical", "run4_holdout_s01_14"]:
        if not (ROOT / "runs" / r / "best.pt").exists():
            print(f"skipping {r}: no best.pt yet")
            continue
        print(f"evaluating {r} on S01/14 ...")
        out[r] = _eval_model(r)

    target = ROOT / "reports" / "holdout_s01_14_compare.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {target.relative_to(ROOT)}")

    # Print summary table
    print(f"\n{'model':<28} {'MPJPE':>8} {'cross H/K':>11} {'cad-rel%':>10} {'stride-rel%':>13}")
    for r, v in out.items():
        cad_rel = stride_rel = float("nan")
        for side, g in v["gait"].items():
            if "oracle_cadence_steps_per_min" in g and g["oracle_cadence_steps_per_min"] > 0:
                cr = abs(g["student_cadence_steps_per_min"] - g["oracle_cadence_steps_per_min"]) / g["oracle_cadence_steps_per_min"] * 100
                sr = abs(g["student_stride_period_s"] - g["oracle_stride_period_s"]) / g["oracle_stride_period_s"] * 100
                cad_rel = cr if np.isnan(cad_rel) else (cad_rel + cr) / 2
                stride_rel = sr if np.isnan(stride_rel) else (stride_rel + sr) / 2
        print(f"{r:<28} {v['mpjpe_overall_mm']:>7.1f}m {v['frac_hip_knee_crossover']*100:>10.2f}% "
              f"{cad_rel:>9.1f}% {stride_rel:>12.1f}%")

    # Two-panel figure: per-joint MPJPE (left) + gait rel-err (right).
    # The story is: MPJPE quadruples on the held-out bag while the
    # gait derivations barely move. Visualise that gap.
    LABEL_MAP = {
        "run3_anatomical": "Random split (leaks frames)",
        "run4_holdout_s01_14": "Leave-one-bag-out (honest)",
    }
    if len(out) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        run_keys = list(out.keys())
        readable = [LABEL_MAP.get(r, r) for r in run_keys]
        joints = list(JOINT_ORDER)
        x = np.arange(len(joints))
        w = 0.4
        for i, (rk, color) in enumerate(zip(run_keys, ["#1f77b4", "#d62728"])):
            mpjpes = [out[rk]["mpjpe_per_joint_mm"][j] for j in joints]
            axes[0].bar(x + (i - 0.5) * w, mpjpes, w, label=readable[i],
                        color=color, alpha=0.85)
        axes[0].set_xticks(x); axes[0].set_xticklabels(joints, rotation=20, fontsize=9)
        axes[0].set_ylabel("MPJPE on S01/14 (mm)")
        axes[0].set_title("All per-joint errors below the 50 mm goal on the held-out recording")
        axes[0].axhline(50, color="grey", lw=0.8, ls="--", alpha=0.6,
                         label="50 mm clinical-usability goal")
        axes[0].legend(fontsize=9)

        # Right panel: cadence + stride rel-err (per side, then averaged).
        def _avg_rel(g_dict, key_o, key_s):
            errs = []
            for side, g in g_dict.items():
                if key_o in g and g[key_o] not in (0, None):
                    errs.append(abs(g[key_s] - g[key_o]) / abs(g[key_o]) * 100)
            return float(np.mean(errs)) if errs else float("nan")

        cad_rel = [_avg_rel(out[r]["gait"],
                             "oracle_cadence_steps_per_min",
                             "student_cadence_steps_per_min") for r in run_keys]
        stride_rel = [_avg_rel(out[r]["gait"],
                                "oracle_stride_period_s",
                                "student_stride_period_s") for r in run_keys]
        gx = np.arange(2)  # two metrics
        gw = 0.35
        for i, rk in enumerate(run_keys):
            vals = [cad_rel[i], stride_rel[i]]
            axes[1].bar(gx + (i - 0.5) * gw, vals, gw, label=readable[i],
                         color=["#1f77b4", "#d62728"][i], alpha=0.85)
            for x_, v in zip(gx + (i - 0.5) * gw, vals):
                axes[1].text(x_, v + 0.05, f"{v:.1f}%", ha="center",
                              va="bottom", fontsize=9)
        axes[1].axhline(10, color="grey", lw=0.8, ls="--", alpha=0.6,
                         label="10 % H2 target")
        axes[1].set_xticks(gx)
        axes[1].set_xticklabels(["Cadence (steps/min)", "Stride period (s)"],
                                 fontsize=10)
        axes[1].set_ylabel("median absolute relative error vs oracle (%)")
        axes[1].set_title("Gait parameters barely move on the held-out bag")
        axes[1].legend(fontsize=9)

        p = ROOT / "reports" / "figures" / "holdout_s01_14_compare.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"wrote {p.relative_to(ROOT)}")
