# Readout prediction metric — drumjepa_v1_e40 (epoch 19, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.342 [0.316, 0.358] | 0.15 | 0.989 |
| ceiling | 32000 | 0.338 [0.312, 0.354] | 0.15 | 0.977 |
| full_mask | 32000 | 0.290 [0.259, 0.309] | 0.05 | 0.925 |

Onset F1 of the predicted grid as a fraction of the ceiling: **1.011**; kit accuracy as a fraction of the ceiling: 1.012.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.343 against a ceiling of 0.340 on the same steps (ratio 1.011).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.2120, and the teacher's mean per-dim variance is 1.5880 (normalized error 0.1335).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.582 | 0.583 |
| snare | 0.554 | 0.553 |
| rimshot | 0.630 | 0.620 |
| crossstick | 0.158 | 0.144 |
| tom_hi | 0.233 | 0.225 |
| tom_mid | 0.111 | 0.109 |
| tom_lo | 0.209 | 0.205 |
| hh_closed | 0.743 | 0.737 |
| hh_open | 0.183 | 0.180 |
| hh_pedal | 0.391 | 0.390 |
| crash1 | 0.201 | 0.217 |
| crash2 | 0.057 | 0.052 |
| ride | 0.621 | 0.610 |
| ride_bell | 0.118 | 0.115 |
