# Readout prediction metric — drumjepa_v1_melrec_s2 (epoch 19, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.519 [0.495, 0.531] | 0.20 | 0.998 |
| ceiling | 32000 | 0.551 [0.525, 0.563] | 0.25 | 0.996 |
| full_mask | 32000 | 0.494 [0.469, 0.505] | 0.20 | 0.998 |

Onset F1 of the predicted grid as a fraction of the ceiling: **0.943**; kit accuracy as a fraction of the ceiling: 1.002.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.520 against a ceiling of 0.552 on the same steps (ratio 0.941).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.0423, and the teacher's mean per-dim variance is 1.2209 (normalized error 0.0346).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.793 | 0.798 |
| snare | 0.674 | 0.689 |
| rimshot | 0.715 | 0.755 |
| crossstick | 0.226 | 0.297 |
| tom_hi | 0.533 | 0.572 |
| tom_mid | 0.485 | 0.532 |
| tom_lo | 0.648 | 0.687 |
| hh_closed | 0.805 | 0.810 |
| hh_open | 0.341 | 0.376 |
| hh_pedal | 0.477 | 0.478 |
| crash1 | 0.187 | 0.228 |
| crash2 | 0.168 | 0.262 |
| ride | 0.833 | 0.833 |
| ride_bell | 0.386 | 0.392 |
