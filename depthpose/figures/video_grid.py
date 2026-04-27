"""1×2 video-grid figure: best (in-distribution) and held-out, mid-frame stills.

Pulls one mid-frame still from each of the two side-by-side rendered
videos and lays them out as a single PNG with per-panel captions
(MPJPE in mm). Used in §7 of the blog as the static fallback for the
embedded ``<video>`` tags in the HTML build.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(add_completion=False, help=__doc__)

PANELS = [
    # (video filename, blog-friendly label, MPJPE label string)
    ("S01_7.mp4",          "Best case: rs_blue_car_light_change",
        "8 mm MPJPE (random per-session split)"),
    ("S01_14_holdout.mp4", "Held-out: rs_up_incline (S01/14)",
        "40 mm MPJPE (held out from training)"),
]


def _mid_frame(path: Path):
    v = cv2.VideoCapture(str(path))
    n = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    v.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = v.read()
    v.release()
    if not ok:
        raise RuntimeError(f"could not read mid-frame from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@app.command()
def main() -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6 * len(PANELS), 7.5),
                              constrained_layout=True)
    if len(PANELS) == 1:
        axes = [axes]
    for ax, (fname, label, mpjpe_label) in zip(axes, PANELS):
        path = ROOT / "reports" / "videos" / fname
        if not path.exists():
            raise FileNotFoundError(f"missing video: {path}")
        ax.imshow(_mid_frame(path))
        ax.set_title(f"{label}\n{mpjpe_label}", fontsize=12)
        ax.axis("off")
    out = ROOT / "reports" / "figures" / "video_grid.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    app()
