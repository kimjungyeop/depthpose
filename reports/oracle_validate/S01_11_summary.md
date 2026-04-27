# Oracle validation summary — S01/11

- frames: **317**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 16 / 317 | 0.050 |
| knees + ankles valid (gait-relevant) | 317 / 317 | 1.000 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.936 | 1.000 | 0.000 | 0.766 |
| left_hip | 0.839 | 0.095 | 0.000 | 0.707 |
| left_knee | 0.888 | 1.000 | 0.000 | 0.763 |
| right_ankle | 0.933 | 1.000 | 0.000 | 0.752 |
| right_hip | 0.866 | 0.069 | 0.000 | 0.692 |
| right_knee | 0.881 | 1.000 | 0.000 | 0.749 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).