# Readout prediction metric — drumjepa_v1_aojepa (epoch 19, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.370 [0.341, 0.384] | 0.15 | 0.977 |
| ceiling | 32000 | 0.383 [0.353, 0.396] | 0.15 | 0.962 |
| full_mask | 32000 | 0.246 [0.214, 0.267] | 0.05 | 0.836 |

Onset F1 of the predicted grid as a fraction of the ceiling: **0.968**; kit accuracy as a fraction of the ceiling: 1.016.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.371 against a ceiling of 0.384 on the same steps (ratio 0.967).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.1264, and the teacher's mean per-dim variance is 1.0611 (normalized error 0.1191).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.601 | 0.608 |
| snare | 0.572 | 0.577 |
| rimshot | 0.632 | 0.643 |
| crossstick | 0.175 | 0.177 |
| tom_hi | 0.301 | 0.311 |
| tom_mid | 0.150 | 0.213 |
| tom_lo | 0.273 | 0.293 |
| hh_closed | 0.756 | 0.759 |
| hh_open | 0.195 | 0.212 |
| hh_pedal | 0.409 | 0.412 |
| crash1 | 0.162 | 0.160 |
| crash2 | 0.079 | 0.087 |
| ride | 0.687 | 0.693 |
| ride_bell | 0.194 | 0.212 |
