# Oracle validation summary — S01/3

- frames: **359**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 87 / 359 | 0.242 |
| knees + ankles valid (gait-relevant) | 314 / 359 | 0.875 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.862 | 0.997 | 0.017 | 0.662 |
| left_hip | 0.734 | 0.373 | 0.064 | 0.612 |
| left_knee | 0.717 | 1.000 | 0.084 | 0.605 |
| right_ankle | 0.854 | 1.000 | 0.022 | 0.651 |
| right_hip | 0.760 | 0.490 | 0.070 | 0.605 |
| right_knee | 0.732 | 1.000 | 0.097 | 0.592 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).