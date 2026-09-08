# Readout prediction metric — drumjepa_v1_melrec_s1 (epoch 19, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.530 [0.502, 0.544] | 0.20 | 0.998 |
| ceiling | 32000 | 0.552 [0.526, 0.567] | 0.25 | 0.995 |
| full_mask | 32000 | 0.501 [0.476, 0.512] | 0.20 | 0.996 |

Onset F1 of the predicted grid as a fraction of the ceiling: **0.959**; kit accuracy as a fraction of the ceiling: 1.003.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.530 against a ceiling of 0.555 on the same steps (ratio 0.956).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.0560, and the teacher's mean per-dim variance is 1.2026 (normalized error 0.0466).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.775 | 0.788 |
| snare | 0.682 | 0.700 |
| rimshot | 0.724 | 0.732 |
| crossstick | 0.258 | 0.298 |
| tom_hi | 0.551 | 0.568 |
| tom_mid | 0.487 | 0.532 |
| tom_lo | 0.653 | 0.667 |
| hh_closed | 0.800 | 0.804 |
| hh_open | 0.387 | 0.409 |
| hh_pedal | 0.460 | 0.470 |
| crash1 | 0.233 | 0.241 |
| crash2 | 0.163 | 0.241 |
| ride | 0.838 | 0.841 |
| ride_bell | 0.404 | 0.438 |
