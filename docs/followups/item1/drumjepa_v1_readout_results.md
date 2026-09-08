# Readout prediction metric — drumjepa_v1 (epoch 19, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.342 [0.311, 0.358] | 0.10 | 0.992 |
| ceiling | 32000 | 0.342 [0.315, 0.358] | 0.15 | 0.984 |
| full_mask | 32000 | 0.302 [0.272, 0.319] | 0.10 | 0.845 |

Onset F1 of the predicted grid as a fraction of the ceiling: **0.999**; kit accuracy as a fraction of the ceiling: 1.008.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.343 against a ceiling of 0.343 on the same steps (ratio 1.002).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.0919, and the teacher's mean per-dim variance is 1.2137 (normalized error 0.0757).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.569 | 0.578 |
| snare | 0.526 | 0.548 |
| rimshot | 0.581 | 0.593 |
| crossstick | 0.161 | 0.147 |
| tom_hi | 0.246 | 0.214 |
| tom_mid | 0.200 | 0.148 |
| tom_lo | 0.284 | 0.240 |
| hh_closed | 0.708 | 0.737 |
| hh_open | 0.165 | 0.161 |
| hh_pedal | 0.357 | 0.392 |
| crash1 | 0.134 | 0.158 |
| crash2 | 0.081 | 0.085 |
| ride | 0.607 | 0.619 |
| ride_bell | 0.169 | 0.172 |
