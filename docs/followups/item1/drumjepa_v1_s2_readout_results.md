# Readout prediction metric — drumjepa_v1_s2 (epoch 19, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.390 [0.365, 0.402] | 0.15 | 0.995 |
| ceiling | 32000 | 0.386 [0.358, 0.401] | 0.15 | 0.989 |
| full_mask | 32000 | 0.308 [0.278, 0.325] | 0.05 | 0.928 |

Onset F1 of the predicted grid as a fraction of the ceiling: **1.009**; kit accuracy as a fraction of the ceiling: 1.006.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.391 against a ceiling of 0.387 on the same steps (ratio 1.011).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.1094, and the teacher's mean per-dim variance is 1.1215 (normalized error 0.0976).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.583 | 0.586 |
| snare | 0.559 | 0.557 |
| rimshot | 0.666 | 0.651 |
| crossstick | 0.130 | 0.133 |
| tom_hi | 0.334 | 0.309 |
| tom_mid | 0.176 | 0.227 |
| tom_lo | 0.292 | 0.302 |
| hh_closed | 0.749 | 0.749 |
| hh_open | 0.261 | 0.250 |
| hh_pedal | 0.409 | 0.406 |
| crash1 | 0.204 | 0.214 |
| crash2 | 0.111 | 0.078 |
| ride | 0.731 | 0.713 |
| ride_bell | 0.254 | 0.233 |
