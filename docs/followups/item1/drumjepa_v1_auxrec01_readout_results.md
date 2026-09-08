# Readout prediction metric — drumjepa_v1_auxrec01 (epoch 19, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.614 [0.593, 0.626] | 0.25 | 0.995 |
| ceiling | 32000 | 0.609 [0.589, 0.618] | 0.30 | 0.995 |
| full_mask | 32000 | 0.563 [0.534, 0.582] | 0.15 | 0.994 |

Onset F1 of the predicted grid as a fraction of the ceiling: **1.008**; kit accuracy as a fraction of the ceiling: 1.000.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.616 against a ceiling of 0.611 on the same steps (ratio 1.008).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.0655, and the teacher's mean per-dim variance is 1.8030 (normalized error 0.0364).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.916 | 0.901 |
| snare | 0.719 | 0.730 |
| rimshot | 0.807 | 0.780 |
| crossstick | 0.297 | 0.315 |
| tom_hi | 0.624 | 0.612 |
| tom_mid | 0.535 | 0.526 |
| tom_lo | 0.776 | 0.770 |
| hh_closed | 0.838 | 0.842 |
| hh_open | 0.535 | 0.529 |
| hh_pedal | 0.530 | 0.536 |
| crash1 | 0.403 | 0.362 |
| crash2 | 0.227 | 0.280 |
| ride | 0.872 | 0.870 |
| ride_bell | 0.522 | 0.474 |
