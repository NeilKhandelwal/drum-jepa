# Readout prediction metric — drumjepa_v1_actiononly (epoch 19, test split, 6 train kits)

Readouts fit on 29988 train-split clips of the model's own teacher encoder, applied to 4000 test transitions, K=4 masks each at 75% masked, seed 0. Nothing is divided by the teacher's variance: both readouts are frozen linear maps into onset and kit labels.

| condition | n rows | onset macro-F1 | threshold | kit accuracy |
|---|---|---|---|---|
| predicted | 128000 | 0.325 [0.297, 0.340] | 0.10 | 0.993 |
| ceiling | 32000 | 0.330 [0.302, 0.345] | 0.15 | 0.986 |
| full_mask | 32000 | 0.256 [0.237, 0.269] | 0.05 | 0.117 |

Onset F1 of the predicted grid as a fraction of the ceiling: **0.983**; kit accuracy as a fraction of the ceiling: 1.007.

Restricted to the 117233 time steps where more than 50% of the 16 frequency tokens were predicted, the predicted grid scores 0.326 against a ceiling of 0.332 on the same steps (ratio 0.982).

## Protocol

- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step vectors (256-d), then 14 independent logistic regressions against "was this class struck in this 250 ms step", fit on the 12852 train-split clips that belong to the 6 train kits (E5's protocol). The threshold is picked per condition on the validation split (4000 transitions) and applied to test.
- Kit readout: a 14-way logistic regression on the clip embedding (mean over the 128 tokens), fit on all 29988 train-split clips across 14 kits (E3's 14-way probe). Scored transitions come from the train kits only, as in E1 and E2, so chance is 1/14 by class count.
- The readout's training clips are the real, unmasked x_{t+1} of each train-split transition, encoded by the EMA teacher. No prediction is involved in fitting.
- For reference, the E2-style masked prediction error on the same transitions is 0.1385, and the teacher's mean per-dim variance is 1.3049 (normalized error 0.1062).

### Per-class onset F1, predicted grid against the ceiling

| class | predicted | ceiling |
|---|---|---|
| kick | 0.567 | 0.575 |
| snare | 0.524 | 0.549 |
| rimshot | 0.568 | 0.586 |
| crossstick | 0.136 | 0.122 |
| tom_hi | 0.262 | 0.231 |
| tom_mid | 0.183 | 0.160 |
| tom_lo | 0.291 | 0.269 |
| hh_closed | 0.717 | 0.745 |
| hh_open | 0.144 | 0.141 |
| hh_pedal | 0.356 | 0.397 |
| crash1 | 0.088 | 0.098 |
| crash2 | 0.056 | 0.042 |
| ride | 0.547 | 0.590 |
| ride_bell | 0.107 | 0.119 |
