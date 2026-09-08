# Readout prediction metric — drumjepa_v1_auxrec01_s2 (epoch 19, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.695 [0.674, 0.707] | 0.30 | 0.997 |
| ceiling | 32000 | 0.683 [0.660, 0.696] | 0.30 | 0.995 |
| full_mask | 32000 | 0.686 [0.659, 0.701] | 0.20 | 0.996 |

Onset F1 of the predicted grid as a fraction of the ceiling: **1.017**; kit accuracy as a fraction of the ceiling: 1.002.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.697 against a ceiling of 0.685 on the same steps (ratio 1.017).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.0566, and the teacher's mean per-dim variance is 1.9208 (normalized error 0.0294).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.920 | 0.907 |
| snare | 0.845 | 0.812 |
| rimshot | 0.898 | 0.875 |
| crossstick | 0.512 | 0.504 |
| tom_hi | 0.780 | 0.740 |
| tom_mid | 0.742 | 0.734 |
| tom_lo | 0.871 | 0.857 |
| hh_closed | 0.843 | 0.841 |
| hh_open | 0.552 | 0.547 |
| hh_pedal | 0.557 | 0.551 |
| crash1 | 0.395 | 0.423 |
| crash2 | 0.344 | 0.311 |
| ride | 0.880 | 0.880 |
| ride_bell | 0.591 | 0.586 |
