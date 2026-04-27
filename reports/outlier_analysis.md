# Outlier analysis — what drives the 22.5 mm average MPJPE

The Phase 2 student hits **MPJPE 22.5 mm** averaged across the test split, comfortably under the H1 target of <35 mm. But the average hides a sharp split: 12 of 14 sessions are 9–28 mm and **two outlier bags account for nearly all of the gap to the floor at ~17 mm**.

## Per-session test-split MPJPE (run1_baseline `best.pt`)

| session | bag | frames | MPJPE (mm) | dom. axis |
|---|---|---:|---:|---|
| S01/3 | `realsense_vertical_3` | 71 | **82.4** | z (64 mm) |
| S01/5 | `realsense_vertical_5` | 41 | **51.1** | x (36 mm) |
| S01/4 | `realsense_vertical_4` | 121 | 27.5 | x |
| S01/13 | `rs_kitchen_to_deck` | 215 | 21.8 | y/z balanced |
| S01/6 | `rs_along_and_down_incline` | 126 | 19.1 | balanced |
| S01/14 | `rs_up_incline` | 80 | 16.9 | y |
| S01/1 | `realsense_vertical_1` | 76 | 16.6 | balanced |
| S01/10 | `rs_fast_circle_varying_bg` | 75 | 15.8 | balanced |
| S01/12 | `rs_kitchen_loop` | 232 | 15.7 | balanced |
| S01/8 | `rs_bumpy_stone_walk` | 89 | 15.4 | y/right_hip |
| S01/9 | `rs_dark_to_light` | 86 | 15.3 | balanced |
| S01/2 | `realsense_vertical_2` | 69 | 14.2 | balanced |
| S01/11 | `rs_gravel_driveway` | 63 | 11.6 | balanced |
| S01/7 | `rs_blue_car_light_change` | 77 | **8.8** | balanced |

If S01/3 and S01/5 are excluded the test-set mean MPJPE drops to ~**17 mm**.

## Two distinct failure modes

| session | mean \|dx\| | mean \|dy\| | mean \|dz\| | comment |
|---|---:|---:|---:|---|
| S01/3 | 16 | 25 | **64** | z-dominated — depth-quality failure |
| S01/5 | **36** | 14 | 15 | x-dominated — lateral offset |
| (median good session) | ~6 | ~10 | ~7 | balanced |

- **S01/3 — depth-quality failure.** All-axis errors are elevated but the depth axis is ~5× the rest. Visual inspection ([reports/outlier_worst_frames.png](outlier_worst_frames.png) top row) shows a low-contrast scene with dark legs against a dark background; depth-from-IR is unreliable on dark/low-IR-reflectivity surfaces. The oracle's strict frame-eligibility on this bag was already only 24% (vs ~95% typical).
- **S01/5 — lateral offset.** The x-axis dominates by 2.5×. Frames have the subject far from the camera (median z = 0.89 m, the longest in the test set besides S01/4). At that distance a small angular bias becomes a sizeable lateral error in metres. The model has likely overfit to the typical "subject directly ahead at ~0.7 m" geometry.

These are honest dataset-difficulty artifacts, not bugs in the model or pipeline. Both bags should remain in the test split and the blog should report both the headline MPJPE 22.5 mm and the without-outliers MPJPE 17 mm.

## Knee-flex range "outlier" is a denominator artifact

The H2 comparison flagged `knee_flexion_range_deg` with median rel err 16.7% and p90 152% — much worse than the other gait parameters. Drilling in:

- 10 of 28 sides have **oracle range <30°** (subjects barely flex their knees). On these tiny denominators, even modest absolute errors (~10°) blow up to >100% relative.
- On the 18 sides with **oracle range ≥30°** (i.e. real walking with visible flexion), median rel err is **8.8%** and p90 is **45%** — well-behaved.

For the blog, report knee-flex MAX (already 0.3% median / 1.8% p90 — the tightest gait metric) and quote knee-flex RANGE only for the real-flexion subset.

## Implications for the blog

- Lead with MPJPE 22.5 mm overall + 17 mm with the two outlier bags excluded. Be honest about both numbers.
- Use the per-session table as a robustness chart — most sessions cluster tightly around 15 mm, with a long tail driven by depth-quality and distance.
- Knee-flex MAX is the strongest gait-derivation result (0.3% median rel err); flexion RANGE is meaningful only for sides with substantive flexion.
