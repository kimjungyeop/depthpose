# Oracle validation summary — S01/4

- frames: **628**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 355 / 628 | 0.565 |
| knees + ankles valid (gait-relevant) | 613 / 628 | 0.976 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.914 | 1.000 | 0.003 | 0.972 |
| left_hip | 0.818 | 0.696 | 0.027 | 0.843 |
| left_knee | 0.874 | 1.000 | 0.019 | 0.900 |
| right_ankle | 0.906 | 1.000 | 0.003 | 0.986 |
| right_hip | 0.836 | 0.768 | 0.021 | 0.847 |
| right_knee | 0.872 | 1.000 | 0.003 | 0.909 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).