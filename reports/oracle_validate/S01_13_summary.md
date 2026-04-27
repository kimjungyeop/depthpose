# Oracle validation summary — S01/13

- frames: **1126**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 109 / 1126 | 0.097 |
| knees + ankles valid (gait-relevant) | 1006 / 1126 | 0.893 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.864 | 0.999 | 0.024 | 0.690 |
| left_hip | 0.755 | 0.196 | 0.061 | 0.665 |
| left_knee | 0.771 | 0.994 | 0.083 | 0.678 |
| right_ankle | 0.871 | 0.999 | 0.011 | 0.676 |
| right_hip | 0.800 | 0.191 | 0.034 | 0.633 |
| right_knee | 0.770 | 0.993 | 0.031 | 0.663 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).