# Oracle validation summary — S01/1

- frames: **386**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 12 / 386 | 0.031 |
| knees + ankles valid (gait-relevant) | 377 / 386 | 0.977 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.871 | 1.000 | 0.013 | 0.581 |
| left_hip | 0.724 | 0.070 | 0.044 | 0.511 |
| left_knee | 0.748 | 1.000 | 0.023 | 0.551 |
| right_ankle | 0.874 | 1.000 | 0.000 | 0.546 |
| right_hip | 0.808 | 0.168 | 0.010 | 0.496 |
| right_knee | 0.788 | 1.000 | 0.000 | 0.546 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).