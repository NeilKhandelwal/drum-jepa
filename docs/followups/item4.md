# Item 4 — E4 inverse recovery on the aux_rec 0.1 model

Follow-up 4 of docs/followups.md, run 2026-09-07. E5's onset probe reads the same
frequency-pooled state tokens the aux_rec head trains on, so its rise at weight 0.1
is close to tautological. E4 is a different decoder on un-pooled tokens, and it is
the eval that found the original problem, so it is the one that decides whether
action content really came back. It was rerun at weight 1.0 but never at 0.1.

Three seeds of drumjepa_v1_auxrec01 were evaluated with the E4 protocol unchanged:
a 3.5M-param inverse model h(s_t, s_{t+1}[, a_t]) on frozen teacher tokens, trained
8 epochs on 20k transitions, threshold tuned for macro F1 on the validation train
kits and applied unchanged to both test kit groups. Every control is quoted from
docs/experiments.md and docs/optionA/; none was retrained.

Command, once per seed:

    scripts/eval_e4.py --drumjepa <run-dir> --reps drumjepa --with-at drumjepa

## Results

Onset macro F1 at the tuned threshold on the test split, 95% CI from a cluster
bootstrap over sequences. Velocity MAE is over matched onsets in MIDI units, test
split, train kits. The +a_t column is the same h given the previous action.

| representation | train-kit macro F1 [95% CI] | held-out macro F1 | velocity MAE | +a_t train / held-out |
|---|---|---|---|---|
| random-init encoder | 0.251 [0.235, 0.259] | 0.215 | 17.1 | — |
| raw mel, same patching | 0.250 [0.236, 0.256] | 0.221 | 17.5 | 0.201 / — |
| AO-JEPA | 0.164 [0.150, 0.170] | 0.165 | 21.4 | — |
| drum-JEPA (aux_rec 0), seed 0 | 0.118 [0.102, 0.126] | 0.119 | 23.6 | 0.126 / 0.129 |
| aux_rec 1.0, seed 0 | 0.464 [0.444, 0.481] | 0.382 | 19.2 | 0.463 / 0.385 |
| aux_rec 0.1, seed 0 | 0.440 [0.418, 0.458] | 0.354 | 14.8 | 0.445 / 0.362 |
| aux_rec 0.1, seed 1 | 0.392 [0.375, 0.401] | 0.270 | 16.3 | 0.382 / 0.277 |
| aux_rec 0.1, seed 2 | 0.388 [0.370, 0.400] | 0.275 | 17.0 | 0.378 / 0.280 |
| **aux_rec 0.1, mean** | **0.407** | **0.300** | **16.0** | 0.402 / 0.306 |

Per-seed results and per-class breakdowns: docs/followups/item4/.

## Verdict

Passed. The pass criterion was E4 onset F1 above both controls, random-init 0.251
and raw mel 0.250. Every seed clears it on train kits by a wide margin: 0.440,
0.392, 0.388 against 0.251, with confidence intervals that do not come close to
overlapping. The lowest aux_rec 0.1 seed is 1.5 times the better control, and the
whole group is more than three times the unregularized model's 0.118.

Held-out kits pass too, though by less. All three seeds beat both controls there
(0.270 to 0.354 against 0.215 and 0.221), but the margin narrows from a factor of
1.6 at seed 0 to 1.25 at seed 1, and the seed spread on held-out kits (0.084) is
larger than on train kits (0.052). The direction is safe; the magnitude is not.

Velocity is the cleanest result in the table. At weight 0.1 the inverse model
recovers velocity better than any other representation tried, including the two
controls: 14.8 to 17.0 MIDI units of MAE against 17.1 for random-init and 17.5 for
raw mel, and against 23.6 for the unregularized model. Weight 1.0 does not do this
(19.2), so the improvement belongs to the small weight, not to the regularizer in
general.

E4 therefore confirms E5 rather than merely echoing it. A decoder that never sees
the pooled tokens the aux head trains on still recovers substantially more of the
action from the 0.1 model's state than from the unregularized one. The tautology
worry does not hold.

## Caveats

The E4 decoder is small and its 8-epoch budget is fixed, so absolute F1s are far
below published transcription figures and only the ranking is meaningful. This is
the same caveat E4 has carried since 2026-09-06.

Handing h the previous action changes macro F1 by at most 0.010 in either direction
across the three seeds, which is inside the noise. Groove continuation does not
explain these scores, and the sign of the change is not stable across seeds.

Seed 0 is the outlier on the high side: it beats seeds 1 and 2 by about 0.05 on
train kits and 0.08 on held-out kits, and seed 0 is also the seed that all the
earlier single-seed option A numbers came from. Quote the mean (0.407 train, 0.300
held-out) rather than seed 0 alone.

The controls come from the original E4 run on one seed each. Nothing here says how
much of the 0.15 gap between aux_rec 0.1 and the controls would survive if the
controls were also run three times, but the gap is larger than the aux_rec 0.1 seed
spread, which is the only evidence available on that question.

Weight 1.0 still scores higher than 0.1 on E4 train kits (0.464 against a 0.407
mean). Content is not what 0.1 buys over 1.0; prediction is, and that is measured
elsewhere.
