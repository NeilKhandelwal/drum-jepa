# Item 1 — prediction quality read out in a fixed space

Follow-up 1 of docs/followups.md, run 2026-09-07. The claim that aux_rec 0.1
predicts better rests on error divided by the teacher's per-dimension variance, and
the aux term inflates that variance (teacher std 1.09 to 1.33), so the normalization
is doing real work. This eval removes the division.

## Protocol

scripts/eval_readout.py replaces the variance division with a readout: a linear
probe is trained on the model's own teacher embeddings of real, unmasked x_{t+1}
clips, frozen, and then applied to the grid the predictor produces, so every model
is scored in onset F1 and kit accuracy rather than in its own embedding units. Two
readouts are fit on the train split: an onset probe, which pools the 8x16 token grid
over frequency to 8 step vectors and runs E5's 14 independent logistic regressions
against the per-250 ms onset targets, and a kit probe, which is E3's 14-way logistic
regression on the mean over the 128 tokens. Three conditions are scored on 4000
test-split transitions under E1's masks: `predicted`, f's output at the masked
positions with teacher tokens at the visible ones, which is the metric; `ceiling`,
the all-teacher grid of the true x_{t+1}, which is what the readout can do when
prediction is perfect; and `full_mask`, every token predicted at K=1, as in E2. The
ratio predicted/ceiling is the scale-free number, and nothing is divided by any
model's variance.

Commands, one per model:

    scripts/eval_readout.py --run-dir <run-dir> --split test

## Results

Test split, 4000 transitions, K=4 masks at 75% masked, seed 0. Onset is macro F1
over 14 classes with the threshold picked per condition on validation; the CI is a
cluster bootstrap over sequences. Kit accuracy is 14-way, chance 0.071. E2 error is
the masked MSE on the same transitions and teacher var is that model's mean
per-dimension variance, both shown for reference only.

| model | predicted onset F1 [95% CI] | ceiling | ratio | kit acc, predicted | kit acc, ceiling | full-mask onset F1 | full-mask kit acc | E2 error | teacher var |
|---|---|---|---|---|---|---|---|---|---|
| aux_rec 0, seed 0 | 0.342 [0.311, 0.358] | 0.342 | 0.999 | 0.992 | 0.984 | 0.302 | 0.845 | 0.0919 | 1.214 |
| aux_rec 0, seed 1 | 0.376 [0.348, 0.389] | 0.381 | 0.987 | 0.996 | 0.990 | 0.306 | 0.922 | 0.0734 | 1.013 |
| aux_rec 0, seed 2 | 0.390 [0.365, 0.402] | 0.386 | 1.009 | 0.995 | 0.989 | 0.308 | 0.928 | 0.1094 | 1.122 |
| **aux_rec 0, mean** | **0.369** | 0.370 | 0.998 | 0.994 | 0.988 | **0.305** | 0.898 | 0.0916 | 1.116 |
| aux_rec 0.1, seed 0 | 0.614 [0.593, 0.626] | 0.609 | 1.008 | 0.995 | 0.995 | 0.563 | 0.994 | 0.0655 | 1.803 |
| aux_rec 0.1, seed 1 | 0.646 [0.619, 0.660] | 0.648 | 0.996 | 0.997 | 0.996 | 0.621 | 0.996 | 0.0376 | 1.964 |
| aux_rec 0.1, seed 2 | 0.695 [0.674, 0.707] | 0.683 | 1.017 | 0.997 | 0.995 | 0.686 | 0.996 | 0.0566 | 1.921 |
| **aux_rec 0.1, mean** | **0.652** | 0.647 | 1.007 | 0.996 | 0.995 | **0.623** | 0.995 | 0.0532 | 1.896 |
| action-only | 0.325 [0.297, 0.340] | 0.330 | 0.983 | 0.993 | 0.986 | 0.256 | 0.117 | 0.1385 | 1.305 |
| AO-JEPA | 0.370 [0.341, 0.384] | 0.383 | 0.968 | 0.977 | 0.962 | 0.246 | 0.836 | 0.1264 | 1.061 |

Per-model results and per-class breakdowns: docs/followups/item1/.

## Verdict

Passed, with a narrower reading than the plan intended. The criterion was that aux_rec 0.1 beats aux_rec 0 on every seed under the
fixed readout. It does, on both readouts and under both masking conditions, with no
overlap: predicted onset F1 is 0.614, 0.646, 0.695 against 0.342, 0.376, 0.390, so
the worst 0.1 seed beats the best 0 seed by 0.224, and the confidence intervals are
separated by a wide margin. Under full masking the gap is larger still, 0.563-0.686
against 0.302-0.308, and full-mask kit accuracy separates cleanly too (0.994-0.996
against 0.845-0.928). The prediction half of the headline is not an artifact of the
variance normalization.

But the metric measures something narrower than intended, and this is the important
finding. Every model's predicted/ceiling ratio is within 3% of 1.0, so the predicted
grid scores essentially what the true next-window grid scores under the same
readout. That means f reproduces whatever the readout can see almost perfectly for
all eight models, and the differences in the predicted column are differences in
what the encoder puts into the embedding, not in how well the predictor hits it. The
fixed readout confirms the encoder result cleanly, and says less about prediction
quality than the name suggests. Read the ratio column as the prediction measure and
it does not separate the models at all.

The full-mask condition is the exception, and it is the number to quote. With every
token predicted, no visible teacher token can carry the answer, so full-mask F1
divided by the ceiling measures how much of the readable next-window content the
predictor reproduces on its own. That ratio is 0.88, 0.80, 0.80 for the aux_rec 0
seeds and 0.92, 0.96, 1.00 for aux_rec 0.1, with no overlap; action-only is 0.78
and AO-JEPA 0.64. On kit, the same ratio is 0.86-0.94 for aux_rec 0 and 1.00 for
every aux_rec 0.1 seed. So under the one condition where the readout cannot be
satisfied by leaked tokens, the 0.1 predictor reproduces about 96% of the readable
onset content against about 83% for the baseline. That is a scale-free prediction
result in the direction of the headline, with the E2 caveat that full masking is
out of distribution for both models.

The two baselines behave as their architectures predict, which is a useful check on
the readout. The action-only model, whose f gets no state tokens, scores 0.117 on
full-mask kit accuracy, barely above the 0.071 chance level: with every token masked
it has nothing to say about which kit the clip came from. Its predicted-condition
kit accuracy of 0.993 comes entirely from the visible teacher tokens the K=4 masks
leave in place. AO-JEPA has the lowest ratio of any model (0.968) and the lowest
predicted kit accuracy (0.977), so removing action conditioning costs the predictor
some fidelity, but its onset F1 (0.370) sits with the aux_rec 0 seeds rather than
below them.

## Caveats

The ratio being at ceiling for everyone means this eval cannot rank the models on
prediction quality; it ranks them on encoder content and confirms the predictor is
not the bottleneck for any of them. If a genuinely scale-free prediction metric is
still wanted, it needs a readout that the predictor cannot saturate, which likely
means a harder target than per-250 ms onsets or a higher mask ratio than 75%.

The onset readout is fit on frequency-pooled tokens, which docs/experiments.md
records as hiding about half the onset content. The absolute F1s are therefore low
for every model, exactly as in E5.

Kit accuracy is at ceiling under the K=4 masks for seven of eight models
(0.977-0.997), so only the full-mask column carries information there. Quote the
full-mask kit accuracy, not the predicted one.

The two baselines are one seed each, as they have been since 2026-09-06. The
three-seed claim covers only aux_rec 0 and 0.1.

E2 error and teacher variance are in the table for continuity with the earlier
analysis and are not part of this metric. They still show the problem this eval was
built to route around: aux_rec 0.1's teacher variance is about 1.7 times the
baseline's, which is why the raw errors in that column are not comparable across
rows.
