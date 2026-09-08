# Readout prediction metric — drumjepa_v1_s1 (epoch 19, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.376 [0.348, 0.389] | 0.15 | 0.996 |
| ceiling | 32000 | 0.381 [0.352, 0.396] | 0.15 | 0.990 |
| full_mask | 32000 | 0.306 [0.275, 0.322] | 0.10 | 0.922 |

Onset F1 of the predicted grid as a fraction of the ceiling: **0.987**; kit accuracy as a fraction of the ceiling: 1.006.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.377 against a ceiling of 0.382 on the same steps (ratio 0.986).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.0734, and the teacher's mean per-dim variance is 1.0126 (normalized error 0.0725).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.585 | 0.590 |
| snare | 0.563 | 0.562 |
| rimshot | 0.660 | 0.645 |
| crossstick | 0.128 | 0.138 |
| tom_hi | 0.311 | 0.298 |
| tom_mid | 0.193 | 0.231 |
| tom_lo | 0.315 | 0.321 |
| hh_closed | 0.755 | 0.752 |
| hh_open | 0.202 | 0.216 |
| hh_pedal | 0.407 | 0.406 |
| crash1 | 0.163 | 0.197 |
| crash2 | 0.124 | 0.130 |
| ride | 0.669 | 0.652 |
| ride_bell | 0.193 | 0.200 |
