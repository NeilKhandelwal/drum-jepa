# Readout prediction metric — drumjepa_v2_12k_auxrec01 (epoch 19, test split, 12 train kits)

Readouts fit on 30000 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.669 [0.647, 0.686] | 0.30 | 0.997 |
| ceiling | 32000 | 0.665 [0.641, 0.682] | 0.30 | 0.998 |
| full_mask | 32000 | 0.652 [0.630, 0.668] | 0.20 | 0.995 |

Onset F1 of the predicted grid as a fraction of the ceiling: **1.006**; kit accuracy as a fraction of the ceiling: 0.999.

Restricted to the 117184 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.671 against a ceiling of 0.668 on the same steps (ratio 1.005).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 18000 train-split clips that belong to the 12 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 30000 train-split clips across 20 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/20 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.0379, and the teacher's mean per-dim variance is 3.4895 (normalized error 0.0109).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.912 | 0.894 |
| snare | 0.828 | 0.804 |
| rimshot | 0.891 | 0.865 |
| crossstick | 0.568 | 0.534 |
| tom_hi | 0.753 | 0.734 |
| tom_mid | 0.570 | 0.620 |
| tom_lo | 0.819 | 0.803 |
| hh_closed | 0.851 | 0.849 |
| hh_open | 0.558 | 0.548 |
| hh_pedal | 0.555 | 0.546 |
| crash1 | 0.441 | 0.455 |
| crash2 | 0.183 | 0.235 |
| ride | 0.871 | 0.867 |
| ride_bell | 0.564 | 0.559 |
