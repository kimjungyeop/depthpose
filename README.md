# depthpose

Depth-only 3D lower-body pose tracking on a walker-mounted RealSense D435i.

Solo MIT 6.8300 (Spring 2026) project. The deliverable is a blog post +
reproducible code. We train a lightweight depth-only student
(**MobileNetV2 + 2.5D head, 5.22 M params, ~1.36 GMACs / 2.7 GFLOPs at 192×256, 19.9 MB ONNX**)
against an RGB ViTPose++ oracle (~125 M params, via Hugging Face)
treated as pseudo-ground-truth.

**Headline result.** **39.6 mm MPJPE** on a fully held-out walker
recording the model has never seen (95% bootstrap CI [37.7, 41.3] mm
over n = 401 frames; 5,000 replicates), with cadence within 2.0% and
stride period within 1.9% of the oracle. **0.95 ms** inference on
RTX A5000 (ONNX Runtime CUDA fp32); projected **5–10 ms** on Jetson
Xavier NX with TensorRT FP16. Full report: open `dist/index.html` in a
browser, or read `reports/blog.md`.

## Setup

This project pins Python 3.11. Ubuntu 24.04 does not ship 3.11 by
default. Install it first via deadsnakes or pyenv, then:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip pip-tools
pip-compile requirements.in -o requirements.txt
pip install -r requirements.txt
```

Smoke check:

```bash
python -c "import pyrealsense2, torch; print(torch.cuda.is_available())"
```

## Phase 1 entry points

```bash
# Inspect a bag's streams, intrinsics, and duration
python -m depthpose.data.inspect_bag --bag raw_data/<x>.bag

# Extract a bag → on-disk session at data/raw/<subject>/<session>/
# All 14 raw_data/ bags need 270° CW (= 90° CCW) to be upright.
# Default keeps every source frame; --keep-every N to subsample for smoke tests.
python -m depthpose.data.extract_bag --bag raw_data/<x>.bag \
    --subject S01 --session 1 --mount-rotation-cw-deg 270 --out-dir data/raw

# Run the oracle and write 3D keypoint cache
python -m depthpose.oracle.run --session data/raw/S01/1 \
    --out data/labels/S01/1.parquet

# Visual validation contact sheet
python -m depthpose.oracle.validate --session data/raw/S01/1 \
    --labels data/labels/S01/1.parquet
```

## Training, eval, and export

```bash
# Train the canonical leave-one-bag-out model (S01/14 held out)
python -m depthpose.training.train --config configs/holdout_s01_14.yaml

# Evaluate run4 vs run3 on the held-out bag (writes reports/holdout_s01_14_compare.json)
python -m depthpose.eval.holdout_compare

# 5,000-replicate bootstrap CI for the held-out MPJPE
python -m depthpose.eval.bootstrap_ci

# Export ONNX + benchmark latency on CPU/GPU/TensorRT
python -m depthpose.export.export_onnx --run-dir runs/run3_anatomical
python -m depthpose.export.benchmark --run-dir runs/run3_anatomical

# Render side-by-side oracle/student videos with live cadence/stride HUD
python -m depthpose.figures.render_video --session S01/14 \
    --run-dir runs/run4_holdout_s01_14 --out-dir reports/videos

# Build the submission HTML (transcodes videos to H.264 for browser playback)
python -m depthpose.figures.build_html
```

## Layout

```
depthpose/   # data, oracle, model, training, eval, export, figures
configs/     # YAML run configs validated by pydantic (run1..run4)
reports/     # blog.md, supporting JSON, figures, oracle_validate/
dist/        # submission bundle: rendered HTML + figures + 2 demo videos
tests/       # regression-critical units (lift, extract, dataset, loss, gait, ...)
raw_data/    # source .bag files (gitignored)
data/        # extracted sessions + label cache (gitignored)
runs/        # training outputs (gitignored)
```

## Status

- **Phase 1 — data + oracle.** Done. 14 bags extracted, ViTPose++ run on all sessions, contact-sheet validation in `reports/oracle_validate/`. Color-aligned-to-depth at extraction so (u, v) directly indexes the depth frame; hips often land in the depth-FOV gap and are flagged `depth_valid = False` rather than extrapolated. Oracle joints: 6 of COCO-17 lower-body (left/right × hip/knee/ankle).
- **Phase 2 — training.** Done. Four runs:
  - `runs/run1_baseline` (200 ep, pure 3D Smooth-L1, random 80/20 per-frame split) — MPJPE 22.5 mm.
  - `runs/run2_aux2d` (100 ep, + aux 2D heatmap MSE × 0.1) — MPJPE 22.5 mm in half the wall time.
  - `runs/run3_anatomical` (100 ep, + lateral-consistency hinge × 1.0) — MPJPE 21.9 mm; hip-knee crossover 38/7,112 → 10/7,112 (3.8× reduction).
  - **`runs/run4_holdout_s01_14`** (100 ep, leave-one-bag-out: train on S01/1..13, test on S01/14) — **MPJPE 39.6 mm on the held-out bag** (95% bootstrap CI [37.7, 41.3] mm; provenance in `reports/holdout_s01_14_bootstrap.json`). Canonical model for the deployment claim; `runs/run3_anatomical` is the canonical model for the in-distribution ablations.
- **Phase 3 — eval / gait.** Done. On the held-out bag: cadence relative error **2.0 %**, stride period relative error **1.9 %**, hip-knee crossover **0.00 %**, every per-joint MPJPE under the 50 mm clinical-usability goal. See `reports/holdout_s01_14_compare.json`, `reports/h2_compare.json`, `reports/outlier_analysis.md`.
- **Phase 4 — export + benchmark.** Done. `runs/run3_anatomical/student.onnx` (19.9 MB, opset 17, 5.22 M params). Latency on RTX A5000: ONNX-RT CPU 1-thread **24 ms / 41 fps**, ONNX-RT CUDA fp32 **0.95 ms / 1053 fps**, TensorRT FP16 **0.95 ms**. Projected Jetson Xavier NX TRT FP16: **5–10 ms** (~10× faster than the 125 M-param oracle on the same edge silicon).
- **Phase 5 — figures + blog.** Done. Blog at `reports/blog.md`; rendered HTML at `dist/index.html`; figures in `reports/figures/`; two demo videos in `dist/videos/` (`S01_7.mp4` best in-distribution clip, `S01_14_holdout.mp4` the truly held-out bag) with live cadence/stride HUDs.

## Submission

Open `dist/index.html` in a browser. The directory is self-contained
(no internet required). Zip `dist/` for hand-in.
