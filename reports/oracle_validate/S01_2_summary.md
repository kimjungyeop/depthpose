# Oracle validation summary — S01/2

- frames: **346**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 3 / 346 | 0.009 |
| knees + ankles valid (gait-relevant) | 317 / 346 | 0.916 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.867 | 1.000 | 0.000 | 0.538 |
| left_hip | 0.731 | 0.055 | 0.000 | 0.450 |
| left_knee | 0.683 | 1.000 | 0.061 | 0.486 |
| right_ankle | 0.889 | 1.000 | 0.000 | 0.546 |
| right_hip | 0.772 | 0.159 | 0.000 | 0.458 |
| right_knee | 0.703 | 1.000 | 0.046 | 0.494 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).