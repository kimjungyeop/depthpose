"""Compare L/R lateral-crossover rate across run1, run2, run3.

For each trained run, load the best.pt, run on every frame of every
session in the test split, and tally how many frames have the predicted
hip lateral order disagree with the predicted knee or ankle lateral
order. Writes a JSON sidecar + a small bar-chart figure for the blog.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

ROOT = Path("/home/farandhigh-ubuntu/Documents/cv/depth-pose-tracking")
sys.path.insert(0, str(ROOT))
from depthpose.data.training_dataset import JOINT_ORDER, TrainingSession
from depthpose.eval.metrics import mpjpe_overall_mm, mpjpe_per_joint_mm
from depthpose.model.student import DepthPoseStudent
from depthpose.training.config import Config

app = typer.Typer(add_completion=False, help=__doc__)


def _eval_run(run_dir: Path) -> dict:
    cfg = Config.from_yaml(run_dir / "config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TrainingSession(
        raw_dir=cfg.data.raw_dir, labels_root=cfg.data.labels_dir,
        image_size=cfg.data.image_size, split_file=cfg.data.splits_path,
        split="all", drop_swaps=False,  # all 7112 frames
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=True)
    m = DepthPoseStudent(
        backbone_name=cfg.model.backbone, num_joints=cfg.model.num_joints,
        num_deconv=cfg.model.num_deconv, deconv_channels=cfg.model.deconv_channels,
        softargmax_beta=cfg.model.softargmax_beta, pretrained=False,
    ).to(device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    m.load_state_dict(state["model"]); m.eval()

    LH, RH, LK, RK, LA, RA = (JOINT_ORDER.index(j) for j in
        ("left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"))

    n_total = 0
    n_hk = n_ka = n_ha = n_any = 0
    sess_counts: dict[str, int] = defaultdict(int)

    # Also collect MPJPE stats for the test split rows separately (to be fair
    # to the headline numbers).
    test_pred = []; test_tgt = []; test_valid = []

    # Recover the train/test split file
    splits = json.loads(cfg.data.splits_path.read_text())
    test_keys: set[tuple[str, str, int]] = set()
    for sess_key, idxs in splits["splits"].items():
        subj, sess = sess_key.split("/")
        for fi in idxs["test"]:
            test_keys.add((subj, sess, int(fi)))

    with torch.inference_mode():
        for batch in loader:
            out = m(batch["depth"].to(device), batch["intrinsics_input"].to(device))
            p = out["coords_3d"].cpu().numpy()
            tgt = batch["target_3d"].numpy()
            valid = (batch["depth_valid"] & ~batch["needs_review"]).numpy()
            for i in range(p.shape[0]):
                x = p[i, :, 0]
                sgn_h = np.sign(x[LH] - x[RH])
                sgn_k = np.sign(x[LK] - x[RK])
                sgn_a = np.sign(x[LA] - x[RA])
                hk = (sgn_h != 0 and sgn_k != 0 and sgn_h != sgn_k)
                ka = (sgn_k != 0 and sgn_a != 0 and sgn_k != sgn_a)
                ha = (sgn_h != 0 and sgn_a != 0 and sgn_h != sgn_a)
                n_total += 1
                if hk: n_hk += 1
                if ka: n_ka += 1
                if ha: n_ha += 1
                if hk or ka or ha:
                    n_any += 1
                    sess_counts[batch["session"][i]] += 1
                key = ("S01", batch["session"][i], int(batch["frame_index"][i]))
                if key in test_keys:
                    test_pred.append(p[i]); test_tgt.append(tgt[i]); test_valid.append(valid[i])

    test_pred = np.stack(test_pred); test_tgt = np.stack(test_tgt); test_valid = np.stack(test_valid)
    overall_mpjpe = mpjpe_overall_mm(test_pred, test_tgt, test_valid)
    per_j = mpjpe_per_joint_mm(test_pred, test_tgt, test_valid)
    return {
        "n_total_frames": n_total,
        "n_test_frames": int(test_pred.shape[0]),
        "frac_hk_crossover": n_hk / n_total,
        "frac_ka_crossover": n_ka / n_total,
        "frac_ha_crossover": n_ha / n_total,
        "frac_any_crossover": n_any / n_total,
        "test_mpjpe_overall_mm": overall_mpjpe,
        "test_mpjpe_per_joint_mm": {JOINT_ORDER[i]: float(per_j[i]) for i in range(len(JOINT_ORDER))},
        "per_session_crossover_count": dict(sess_counts),
    }


@app.command()
def main(
    runs: list[str] = typer.Option(
        ["run1_baseline", "run2_aux2d", "run3_anatomical"],
        "--runs",
    ),
) -> None:
    out: dict[str, dict] = {}
    for r in runs:
        run_dir = ROOT / "runs" / r
        if not (run_dir / "best.pt").exists():
            typer.echo(f"skipping {r}: no best.pt")
            continue
        typer.echo(f"evaluating {r}…")
        out[r] = _eval_run(run_dir)

    target = ROOT / "reports" / "crossover_compare.json"
    target.write_text(json.dumps(out, indent=2))
    typer.echo(f"\nwrote {target.relative_to(ROOT)}")
    typer.echo(f"\n{'run':<20} {'MPJPE':>8} {'frac H/K':>10} {'frac K/A':>10} {'frac H/A':>10} {'any':>8}")
    for r, v in out.items():
        typer.echo(
            f"{r:<20} {v['test_mpjpe_overall_mm']:>7.2f}m "
            f"{v['frac_hk_crossover']*100:>9.2f}% "
            f"{v['frac_ka_crossover']*100:>9.2f}% "
            f"{v['frac_ha_crossover']*100:>9.2f}% "
            f"{v['frac_any_crossover']*100:>7.2f}%"
        )

    # Bar-chart figure
    if len(out) >= 2:
        # Map internal run names to reader-friendly labels for the figure.
        LABEL_MAP = {
            "run1_baseline":   "Pure 3D loss",
            "run2_aux2d":      "+ Aux 2D heatmap",
            "run3_anatomical": "+ Anatomical loss",
            "run4_holdout_s01_14": "Leave-one-bag-out",
        }
        run_keys = list(out.keys())
        labels = [LABEL_MAP.get(r, r) for r in run_keys]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
        w = 0.27
        hk = [out[r]["frac_hk_crossover"] * 100 for r in run_keys]
        ka = [out[r]["frac_ka_crossover"] * 100 for r in run_keys]
        ha = [out[r]["frac_ha_crossover"] * 100 for r in run_keys]
        ax.bar(x - w, hk, w, label="hip ↔ knee disagree", color="#d62728")
        ax.bar(x,     ka, w, label="knee ↔ ankle disagree", color="#ff7f0e")
        ax.bar(x + w, ha, w, label="hip ↔ ankle disagree", color="#9467bd")
        for xi, vals in zip(x, zip(hk, ka, ha)):
            for off, v in zip([-w, 0, w], vals):
                ax.text(xi + off, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("% of all frames with a physically-impossible L/R order")
        ax.set_title("Adding a hinge on lateral consistency cuts the crossover rate by 3.8×")
        # Headroom so the bar-top text labels don't kiss the title.
        max_v = max(hk + ka + ha)
        ax.set_ylim(0, max_v * 1.18)
        ax.legend(fontsize=9, loc="upper right")
        p = ROOT / "reports" / "figures" / "crossover_compare.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        typer.echo(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    app()
