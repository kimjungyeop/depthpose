# Oracle validation summary — S01/12

- frames: **1166**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 73 / 1166 | 0.063 |
| knees + ankles valid (gait-relevant) | 1153 / 1166 | 0.989 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.920 | 0.999 | 0.000 | 0.772 |
| left_hip | 0.819 | 0.109 | 0.004 | 0.694 |
| left_knee | 0.873 | 1.000 | 0.009 | 0.696 |
| right_ankle | 0.912 | 1.000 | 0.002 | 0.740 |
| right_hip | 0.866 | 0.135 | 0.002 | 0.644 |
| right_knee | 0.863 | 0.999 | 0.003 | 0.680 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).