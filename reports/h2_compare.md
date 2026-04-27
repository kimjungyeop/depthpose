# H2 — gait-parameter robustness

- session-sides matched: **28**
- H2 target: <10% absolute relative error on cadence + stride

| parameter | n | MAE | median rel err (%) | p90 rel err (%) | passes target? |
|---|---:|---:|---:|---:|---|
| Cadence (steps/min) | 28 | 4.364 | 2.8 | 28.4 | ✓ |
| Stride period (s) | 28 | 0.160 | 2.8 | 35.8 | ✓ |
| Step amplitude (m) | 28 | 0.048 | 4.5 | 34.1 | — |
| Knee flex max (°) | 28 | 0.940 | 0.3 | 1.8 | — |
| Knee flex range (°) | 28 | 13.529 | 16.7 | 152.1 | — |