# Readout prediction metric — drumjepa_v1_vcrec (epoch 19, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.247 [0.226, 0.262] | 0.10 | 0.951 |
| ceiling | 32000 | 0.245 [0.224, 0.261] | 0.10 | 0.939 |
| full_mask | 32000 | 0.217 [0.195, 0.235] | 0.05 | 0.236 |

Onset F1 of the predicted grid as a fraction of the ceiling: **1.007**; kit accuracy as a fraction of the ceiling: 1.013.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.248 against a ceiling of 0.246 on the same steps (ratio 1.007).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.1711, and the teacher's mean per-dim variance is 1.7515 (normalized error 0.0977).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.566 | 0.566 |
| snare | 0.506 | 0.506 |
| rimshot | 0.380 | 0.369 |
| crossstick | 0.078 | 0.080 |
| tom_hi | 0.119 | 0.113 |
| tom_mid | 0.057 | 0.061 |
| tom_lo | 0.117 | 0.117 |
| hh_closed | 0.666 | 0.667 |
| hh_open | 0.099 | 0.098 |
| hh_pedal | 0.314 | 0.314 |
| crash1 | 0.103 | 0.095 |
| crash2 | 0.014 | 0.009 |
| ride | 0.359 | 0.364 |
| ride_bell | 0.077 | 0.073 |
