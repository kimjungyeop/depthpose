# Oracle validation summary — S01/6

- frames: **631**

## Frame-eligibility metrics

| metric | count | fraction |
|---|---:|---:|
| all 6 joints valid (strict) | 65 / 631 | 0.103 |
| knees + ankles valid (gait-relevant) | 621 / 631 | 0.984 |

**Use `knees + ankles valid` as the frame-eligibility metric.** Hip drop-outs are a known depth-FOV-gap property and should not disqualify the rest of the frame. Per-joint masking — *never* whole-frame filtering — is the policy for both training loss and evaluation MPJPE (see project memory: per-joint masking).

## Per-joint

| joint | mean conf | frac depth valid | frac needs review | median z (m) |
|---|---:|---:|---:|---:|
| left_ankle | 0.929 | 0.998 | 0.002 | 0.744 |
| left_hip | 0.821 | 0.211 | 0.005 | 0.706 |
| left_knee | 0.866 | 0.998 | 0.000 | 0.670 |
| right_ankle | 0.902 | 1.000 | 0.000 | 0.706 |
| right_hip | 0.840 | 0.149 | 0.013 | 0.786 |
| right_knee | 0.841 | 0.992 | 0.006 | 0.700 |

## Notes
- `conf_2d` is the ViTPose detection score in [0, 1].
- `depth_valid` = depth-median sampling returned a nonzero value (3×3 patch).
- `needs_review` = `conf_2d` < threshold (default 0.5).
- Hip frames where `depth_valid=False` are typically the stride-extension instants when the hip moves into the depth-FOV gap at the top of the frame (see project memory: hip-depth-invalid policy).