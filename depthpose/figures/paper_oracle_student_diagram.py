"""Paper-style oracle-student training diagram.

Outputs:
- reports/figures/oracle_student_training_diagram.svg
- reports/figures/oracle_student_training_diagram.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


INK = "#151515"
MUTED = "#595959"
ORACLE_FILL = "#eef3fb"
ORACLE_EDGE = "#426b9a"
STUDENT_FILL = "#eef6ef"
STUDENT_EDGE = "#3d6b4e"
STORE_FILL = "#fbf4e6"
STORE_EDGE = "#8a6c2f"
LOSS_FILL = "#fcf1ec"
LOSS_EDGE = "#9a4a28"


def box(ax, x, y, w, h, text, *, fc, ec, fontsize=8.7, weight="normal"):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontweight=weight,
    )


def arrow(ax, x1, y1, x2, y2, *, color=MUTED, lw=1.35, ls="solid"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=12,
            shrinkA=3,
            shrinkB=3,
            linestyle=ls,
        ),
    )


def cylinder(ax, x, y, w, h, *, fc, ec):
    cap_h = h * 0.24
    ax.add_patch(
        mpatches.Rectangle(
            (x, y + cap_h / 2),
            w,
            h - cap_h,
            facecolor=fc,
            edgecolor="none",
        )
    )
    ax.plot([x, x], [y + cap_h / 2, y + h - cap_h / 2], color=ec, lw=1.2)
    ax.plot([x + w, x + w], [y + cap_h / 2, y + h - cap_h / 2], color=ec, lw=1.2)
    ax.add_patch(
        mpatches.Ellipse(
            (x + w / 2, y + h - cap_h / 2),
            w,
            cap_h,
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.2,
        )
    )
    ax.add_patch(
        mpatches.Arc(
            (x + w / 2, y + cap_h / 2),
            w,
            cap_h,
            theta1=180,
            theta2=360,
            edgecolor=ec,
            linewidth=1.2,
        )
    )


def build() -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(13.6, 7.7), constrained_layout=True)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 7.7)
    ax.axis("off")

    ax.text(0.2, 7.42, "Oracle-student training pipeline", fontsize=16, fontweight="bold", color=INK)
    ax.text(
        0.2,
        7.14,
        "Offline RGB oracle generates pseudo-3D labels; the deployed student learns from depth alone.",
        fontsize=10,
        color=MUTED,
    )

    ax.text(0.2, 6.48, "A. Offline oracle label generation", fontsize=11, fontweight="bold", color=ORACLE_EDGE)

    y_top = 5.45
    h = 0.8
    top_nodes = [
        (0.25, 1.35, "RGB frame"),
        (1.95, 1.95, "ViTPose++ oracle\n6 lower-body joints"),
        (4.3, 1.55, "2D keypoints\n(u, v), conf"),
        (6.2, 1.95, "Aligned depth lookup\n3x3 median at each pixel"),
        (8.55, 1.8, "Unproject with\ncamera intrinsics"),
        (10.7, 1.65, "3D joint labels\n(X, Y, Z)"),
    ]
    for x, w, label in top_nodes:
        box(ax, x, y_top, w, h, label, fc=ORACLE_FILL, ec=ORACLE_EDGE)
    for i in range(len(top_nodes) - 1):
        arrow(
            ax,
            top_nodes[i][0] + top_nodes[i][1],
            y_top + h / 2,
            top_nodes[i + 1][0],
            y_top + h / 2,
        )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (6.28, 4.78),
            2.36,
            0.42,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#fff4f2",
            edgecolor="#ca7a6b",
            linewidth=0.9,
        )
    )
    ax.text(
        7.46,
        4.99,
        "depth = 0 -> depth_valid = False; write NaN, never extrapolate",
        ha="center",
        va="center",
        fontsize=8.1,
        color="#9d3721",
    )

    pq_x, pq_y, pq_w, pq_h = 4.5, 3.72, 4.55, 0.96
    cylinder(ax, pq_x, pq_y, pq_w, pq_h, fc=STORE_FILL, ec=STORE_EDGE)
    ax.text(pq_x + pq_w / 2, pq_y + 0.62, "Pseudo-label parquet per session", fontsize=10.2, fontweight="bold", color=INK, ha="center")
    ax.text(
        pq_x + pq_w / 2,
        pq_y + 0.34,
        "(u, v), conf_2d, (X, Y, Z), depth_valid, needs_review",
        fontsize=8.45,
        color=MUTED,
        ha="center",
    )
    arrow(ax, 11.52, y_top, 8.1, pq_y + pq_h + 0.02, color=STORE_EDGE)

    ax.text(0.2, 3.1, "B. Depth-only student training", fontsize=11, fontweight="bold", color=STUDENT_EDGE)

    y_bot = 1.95
    bottom_nodes = [
        (0.25, 1.45, "Depth frame\n192 x 256"),
        (2.05, 1.5, "Scaled intrinsics\n(fx, fy, cx, cy)"),
        (3.95, 1.95, "MobileNetV2\n1-channel backbone"),
        (6.3, 1.85, "3x deconv head\nstride 32 -> 4"),
        (8.55, 2.0, "2J outputs:\nheatmaps + z maps"),
        (10.95, 1.85, "Soft-argmax +\nexpected z"),
    ]
    for x, w, label in bottom_nodes:
        box(ax, x, y_bot, w, h, label, fc=STUDENT_FILL, ec=STUDENT_EDGE)

    arrow(ax, 1.7, y_bot + h / 2, 3.95, y_bot + h / 2)
    arrow(ax, 5.9, y_bot + h / 2, 6.3, y_bot + h / 2)
    arrow(ax, 8.15, y_bot + h / 2, 8.55, y_bot + h / 2)
    arrow(ax, 10.55, y_bot + h / 2, 10.95, y_bot + h / 2)
    arrow(ax, 2.8, y_bot + h / 2, 3.95, y_bot + h / 2, color=STUDENT_EDGE)

    pred_x, pred_y, pred_w = 10.95, 0.84, 1.85
    box(ax, pred_x, pred_y, pred_w, h, "Unproject ->\npredicted 3D pose", fc=STUDENT_FILL, ec=STUDENT_EDGE)
    arrow(ax, 11.88, y_bot, 11.88, pred_y + h, color=STUDENT_EDGE)

    loss_x, loss_y, loss_w, loss_h = 4.35, 0.18, 5.15, 1.12
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (loss_x, loss_y),
            loss_w,
            loss_h,
            boxstyle="round,pad=0.03,rounding_size=0.07",
            facecolor=LOSS_FILL,
            edgecolor=LOSS_EDGE,
            linewidth=1.2,
            linestyle="--",
        )
    )
    ax.text(loss_x + 0.18, loss_y + loss_h - 0.24, "Training losses", fontsize=10.2, fontweight="bold", color=LOSS_EDGE, ha="left")
    ax.text(
        loss_x + 0.18,
        loss_y + 0.62,
        "1) Smooth-L1 on 3D joints, masked by depth_valid and not needs_review",
        fontsize=8.45,
        color=INK,
        ha="left",
    )
    ax.text(
        loss_x + 0.18,
        loss_y + 0.37,
        "2) Auxiliary 2D heatmap MSE, skipping low-confidence oracle joints",
        fontsize=8.45,
        color=INK,
        ha="left",
    )
    ax.text(
        loss_x + 0.18,
        loss_y + 0.12,
        "3) Anatomical lateral-consistency hinge on hip, knee, and ankle order",
        fontsize=8.45,
        color=INK,
        ha="left",
    )

    arrow(ax, pq_x + pq_w / 2, pq_y, 6.25, loss_y + loss_h, color=LOSS_EDGE, ls="dashed")
    arrow(ax, pred_x + pred_w / 2, pred_y, 8.55, loss_y + loss_h, color=LOSS_EDGE, ls="dashed")

    ax.text(
        0.25,
        0.22,
        "Split used in the final report: train on S01/1-13 (6711 frames), hold out S01/14 entirely (401 frames); best checkpoint selected by validation MPJPE.",
        fontsize=8.5,
        color=MUTED,
    )

    svg_path = OUT_DIR / "oracle_student_training_diagram.svg"
    png_path = OUT_DIR / "oracle_student_training_diagram.png"
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


if __name__ == "__main__":
    build()
