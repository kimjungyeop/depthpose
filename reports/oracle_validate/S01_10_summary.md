# Oracle validation summary — S01/10

- frames: **385**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 9 / 385 | 0.023 |
| knees + ankles valid (gait-relevant) | 385 / 385 | 1.000 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.926 | 1.000 | 0.000 | 0.694 |
| left_hip | 0.831 | 0.065 | 0.000 | 0.696 |
| left_knee | 0.885 | 1.000 | 0.000 | 0.687 |
| right_ankle | 0.909 | 1.000 | 0.000 | 0.671 |
| right_hip | 0.872 | 0.026 | 0.000 | 0.698 |
| right_knee | 0.881 | 1.000 | 0.000 | 0.679 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).