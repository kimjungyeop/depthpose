# Oracle validation summary — S01/9

- frames: **437**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 15 / 437 | 0.034 |
| knees + ankles valid (gait-relevant) | 433 / 437 | 0.991 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.933 | 1.000 | 0.000 | 0.812 |
| left_hip | 0.834 | 0.066 | 0.009 | 0.712 |
| left_knee | 0.876 | 0.998 | 0.007 | 0.734 |
| right_ankle | 0.922 | 1.000 | 0.000 | 0.750 |
| right_hip | 0.862 | 0.059 | 0.009 | 0.716 |
| right_knee | 0.870 | 1.000 | 0.009 | 0.762 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).