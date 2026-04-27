# Oracle validation summary — S01/5

- frames: **214**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 169 / 214 | 0.790 |
| knees + ankles valid (gait-relevant) | 203 / 214 | 0.949 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.901 | 0.991 | 0.005 | 0.908 |
| left_hip | 0.823 | 0.907 | 0.033 | 0.847 |
| left_knee | 0.833 | 0.991 | 0.019 | 0.850 |
| right_ankle | 0.893 | 1.000 | 0.000 | 0.910 |
| right_hip | 0.824 | 0.925 | 0.037 | 0.854 |
| right_knee | 0.834 | 0.991 | 0.000 | 0.878 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).