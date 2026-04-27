# Oracle validation summary — S01/14

- frames: **401**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 135 / 401 | 0.337 |
| knees + ankles valid (gait-relevant) | 401 / 401 | 1.000 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.956 | 1.000 | 0.000 | 0.903 |
| left_hip | 0.843 | 0.426 | 0.000 | 0.750 |
| left_knee | 0.917 | 1.000 | 0.000 | 0.733 |
| right_ankle | 0.961 | 1.000 | 0.000 | 0.883 |
| right_hip | 0.876 | 0.382 | 0.000 | 0.731 |
| right_knee | 0.910 | 1.000 | 0.000 | 0.733 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).