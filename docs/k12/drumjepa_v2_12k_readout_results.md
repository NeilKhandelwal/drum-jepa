# Readout prediction metric — drumjepa_v2_12k (epoch 19, test split, 12 train kits)

Readouts fit on 30000 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.365 [0.336, 0.378] | 0.15 | 0.997 |
| ceiling | 32000 | 0.393 [0.366, 0.407] | 0.15 | 0.996 |
| full_mask | 32000 | 0.330 [0.301, 0.346] | 0.10 | 0.996 |

Onset F1 of the predicted grid as a fraction of the ceiling: **0.928**; kit accuracy as a fraction of the ceiling: 1.001.

Restricted to the 117184 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.364 against a ceiling of 0.393 on the same steps (ratio 0.926).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 18000 train-split clips that belong to the 12 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 30000 train-split clips across 20 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/20 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.0105, and the teacher's mean per-dim variance is 1.9078 (normalized error 0.0055).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.614 | 0.625 |
| snare | 0.566 | 0.572 |
| rimshot | 0.543 | 0.594 |
| crossstick | 0.142 | 0.173 |
| tom_hi | 0.287 | 0.377 |
| tom_mid | 0.390 | 0.400 |
| tom_lo | 0.416 | 0.489 |
| hh_closed | 0.775 | 0.776 |
| hh_open | 0.200 | 0.222 |
| hh_pedal | 0.391 | 0.387 |
| crash1 | 0.016 | 0.020 |
| crash2 | 0.006 | 0.040 |
| ride | 0.632 | 0.619 |
| ride_bell | 0.134 | 0.212 |
