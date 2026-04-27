# Oracle validation summary — S01/7

- frames: **387**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 7 / 387 | 0.018 |
| knees + ankles valid (gait-relevant) | 386 / 387 | 0.997 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.921 | 1.000 | 0.000 | 0.676 |
| left_hip | 0.801 | 0.028 | 0.000 | 0.697 |
| left_knee | 0.868 | 1.000 | 0.000 | 0.664 |
| right_ankle | 0.917 | 1.000 | 0.000 | 0.668 |
| right_hip | 0.840 | 0.036 | 0.000 | 0.649 |
| right_knee | 0.869 | 1.000 | 0.003 | 0.680 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).