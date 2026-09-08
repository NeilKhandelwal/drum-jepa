# Readout prediction metric — drumjepa_v1_e40 (epoch 39, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.311 [0.285, 0.329] | 0.10 | 0.981 |
| ceiling | 32000 | 0.310 [0.284, 0.329] | 0.10 | 0.984 |
| full_mask | 32000 | 0.269 [0.243, 0.286] | 0.05 | 0.844 |

Onset F1 of the predicted grid as a fraction of the ceiling: **1.002**; kit accuracy as a fraction of the ceiling: 0.997.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.313 against a ceiling of 0.312 on the same steps (ratio 1.002).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.0920, and the teacher's mean per-dim variance is 1.5442 (normalized error 0.0596).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.566 | 0.567 |
| snare | 0.514 | 0.516 |
| rimshot | 0.562 | 0.549 |
| crossstick | 0.140 | 0.139 |
| tom_hi | 0.215 | 0.206 |
| tom_mid | 0.095 | 0.101 |
| tom_lo | 0.198 | 0.191 |
| hh_closed | 0.708 | 0.708 |
| hh_open | 0.153 | 0.152 |
| hh_pedal | 0.348 | 0.348 |
| crash1 | 0.161 | 0.148 |
| crash2 | 0.042 | 0.075 |
| ride | 0.556 | 0.555 |
| ride_bell | 0.098 | 0.090 |
