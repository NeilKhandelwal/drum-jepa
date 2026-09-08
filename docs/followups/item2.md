# Item 2 — control auxiliary target

Follow-up 2 of docs/followups.md, run 2026-09-07. The option A headline is that a
weak action-reconstruction term restores action content to the state and improves
the world model's prediction at the same time. That claim cannot be read as
action-specific until a term with the same shape and weight but a non-action target
has been run. The control is `aux_target: mel` (configs/drumjepa_v1_melrec.yaml):
option A's head and weight 0.1 are kept and only the target changes, from the
current window's drumroll under BCE to the window's own normalized log-mel,
averaged over each step's 25 frames, under MSE. notes/decisions.md
("Control auxiliary target, 2026-09-07") records the weight choice: measured over 8
real training batches at the seed-0 init, the action BCE is 0.967 and the mel MSE is
0.914, within 6% of each other and inside the batch-to-batch spread, so the weight
stays at 0.1 and the two runs differ in the target alone. Everything else — data,
masks, recipe, seed 0, 20 epochs — matches drumjepa_v1_auxrec01. Seeds 1 and 2 were
run afterwards from scripts/run_followups_chain2.sh, with
configs/drumjepa_v1_melrec_s1.yaml and configs/drumjepa_v1_melrec_s2.yaml; each is
identical to the seed-0 config except for the seed and the run name.

## Artifacts and commands

Copied from runs/drumjepa_v1_melrec/ into docs/followups/item2/, prefix
`drumjepa_v1_melrec_`: `train_config.json`, `train_metrics.csv`,
`e1_results.{json,md}`, `e1_win_rates.png`, `e2_results.{json,md}`,
`e5_results.{json,md}`, `e5_f1_by_class.png`, `readout_results.{json,md}`. The same
files without the figures are copied from runs/drumjepa_v1_melrec_s1/ and
runs/drumjepa_v1_melrec_s2/ under the prefixes `drumjepa_v1_melrec_s1_` and
`drumjepa_v1_melrec_s2_`.

Training:

    .venv/bin/python scripts/train.py --config configs/drumjepa_v1_melrec.yaml

Evals, as in scripts/run_followups_chain.sh:

    .venv/bin/python scripts/eval_e1.py --run-dir runs/drumjepa_v1_melrec
    .venv/bin/python scripts/eval_e2.py --full runs/drumjepa_v1_melrec \
        --action-only runs/drumjepa_v1_actiononly
    .venv/bin/python scripts/eval_e5.py --drumjepa runs/drumjepa_v1_melrec --skip aojepa
    .venv/bin/python scripts/eval_readout.py --run-dir runs/drumjepa_v1_melrec --split test

Seeds 1 and 2 run the same training and evals from
configs/drumjepa_v1_melrec_s{1,2}.yaml; scripts/run_followups_chain2.sh does both.

## Results

Seed 0 for every row. The baseline E2 row comes from docs/e2/results_validation.json
and the baseline E5 row from docs/e5/results.json; everything else comes from the run
directories.

Prediction, MSE-based. Errors are masked MSE in each model's own teacher space;
normalized error divides the clean error by that model's squared final teacher std.

| model | teacher std | E2 clean err | E2 kit-swap err | E2 full-mask err | clean err / var |
|---|---|---|---|---|---|
| aux_rec 0 (baseline) | 1.081 | 0.095 | 0.115 | 0.624 | 0.082 |
| action target 0.1 | 1.333 | 0.067 | 1.734 | 0.164 | 0.038 |
| mel target 0.1 (control) | 1.113 | 0.026 | 2.024 | 0.033 | 0.021 |

E1 perturbation win rates, within model, chance 0.5.

| model | kit swap | random state | state shift | random action |
|---|---|---|---|---|
| aux_rec 0 (baseline) | 0.993 | 0.973 | 0.594 | 0.993 |
| action target 0.1 | 0.997 | 0.990 | 0.633 | 1.000 |
| mel target 0.1 (control) | 0.998 | 0.985 | 0.574 | 0.963 |

Content and the item 1 readout. E5 is the linear onset probe on the test split, macro
F1 over 14 classes. Readout columns are onset F1 in the fixed teacher-embedding
readout of item 1; `full-mask / ceiling` is the scale-free prediction measure.

| model | E5 train kits | E5 held-out kits | readout predicted F1 | ceiling | full-mask F1 | full-mask / ceiling | full-mask kit acc |
|---|---|---|---|---|---|---|---|
| aux_rec 0 (baseline) | 0.223 | 0.188 | 0.342 | 0.342 | 0.302 | 0.884 | 0.845 |
| action target 0.1 | 0.582 | 0.396 | 0.614 | 0.609 | 0.563 | 0.925 | 0.994 |
| mel target 0.1 (control) | 0.522 | 0.250 | 0.543 | 0.571 | 0.494 | 0.865 | 0.996 |

E5 controls, unchanged from docs/e5/: random-init encoder 0.365 train kits / 0.320
held-out kits, raw mel 0.583 / 0.277.

Three seeds per group. The baseline and action-target rows come from
docs/optionA/seeds/ and docs/followups/item1.md, the control rows from
runs/drumjepa_v1_melrec{,_s1,_s2}/; the seed-0 rows repeat the tables above. Readout
and E5 are on the test split, E2 and E1 on validation.

Prediction and dynamics.

| group | seed | clean err / var | E2 win vs action-only | E2 kit-swap / clean err | E1 random state | E1 kit swap |
|---|---|---|---|---|---|---|
| aux_rec 0 | 0 | 0.082 | 0.780 | 1.2 | 0.973 | 0.993 |
| aux_rec 0 | 1 | 0.079 | 0.915 | 1.7 | 0.980 | 0.995 |
| aux_rec 0 | 2 | 0.108 | 0.635 | 1.3 | 0.979 | 0.994 |
| action target 0.1 | 0 | 0.038 | 0.931 | 25.8 | 0.990 | 0.997 |
| action target 0.1 | 1 | 0.026 | 0.978 | 60.4 | 0.991 | 0.995 |
| action target 0.1 | 2 | 0.033 | 0.941 | 48.5 | 0.990 | 0.997 |
| mel target 0.1 | 0 | 0.021 | 0.965 | 77.3 | 0.985 | 0.998 |
| mel target 0.1 | 1 | 0.035 | 0.928 | 48.7 | 0.984 | 0.998 |
| mel target 0.1 | 2 | 0.024 | 0.956 | 70.2 | 0.987 | 0.998 |

Content and the readout ratio.

| group | seed | E5 train kits | E5 held-out kits | full-mask / ceiling | ceiling | full-mask kit acc |
|---|---|---|---|---|---|---|
| aux_rec 0 | 0 | 0.223 | 0.188 | 0.884 | 0.342 | 0.845 |
| aux_rec 0 | 1 | 0.265 | 0.208 | 0.802 | 0.381 | 0.922 |
| aux_rec 0 | 2 | 0.263 | 0.209 | 0.796 | 0.386 | 0.928 |
| action target 0.1 | 0 | 0.582 | 0.396 | 0.925 | 0.609 | 0.994 |
| action target 0.1 | 1 | 0.618 | 0.253 | 0.958 | 0.648 | 0.996 |
| action target 0.1 | 2 | 0.664 | 0.304 | 1.003 | 0.683 | 0.996 |
| mel target 0.1 | 0 | 0.522 | 0.250 | 0.865 | 0.571 | 0.996 |
| mel target 0.1 | 1 | 0.500 | 0.263 | 0.907 | 0.552 | 0.996 |
| mel target 0.1 | 2 | 0.505 | 0.248 | 0.897 | 0.551 | 0.998 |

Group means: the readout ratio is 0.827 for the baseline, 0.962 for the action target
and 0.890 for the control; E5 held-out is 0.202, 0.318 and 0.254.

Training curves (drumjepa_v1_melrec_train_metrics.csv, and the auxrec01 metrics for
comparison): the mel aux loss falls from 0.18 at epoch 0 to 0.03 at epoch 2 and
0.013-0.016 at epoch 19, the same in all three control seeds. The action aux loss
stays near 0.3 for the whole run, 0.57 at epoch 0 and 0.31 at epoch 19. Validation
state loss at epoch 19 is 0.098 for the baseline, 0.069 for the action target and
0.027-0.040 for the control's three seeds.

## Verdict

**On the MSE metrics the control wins.** At seed 0 it has the lowest clean error
(0.026 against 0.067 and 0.095) and the lowest full-mask error (0.033 against 0.164
and 0.624). Across three seeds its normalized error is 0.021-0.035, at or below the
action target's 0.026-0.038 and far below the baseline's 0.079-0.108; the two
regularized groups overlap each other and both are separated from the baseline. Read
only through the metrics the option A sweep used, the control refutes the
action-specific claim outright: a regularizer with no action content in its target
predicts at least as well as the action term does.

**Both regularized models moved into a different regime, and the control is what
exposed it.** Kit-swapped error is 26 to 60 times the clean error for the action
target and 49 to 77 times it for the control across three seeds, where the
baseline's is 1.2 to 1.7 times. At the same time the full-mask error collapses toward
the clean error, from 6.6 times it for the baseline at seed 0 to 2.4 times for the
action target and 1.3 times for the control. Both numbers say the same thing: the
predictor leans much harder on s_t, and the target has become close to a function of
s_t plus the action. Lower MSE in this regime means the target is easier to predict,
not that the world model is better. This shift was present in the action-target run
from the start and was not noticed until the control was run against it.

**The scale-free readout orders the three groups where the MSE does not.** On the
full-mask/ceiling ratio from item 1 — the one prediction measure here that cannot be
improved by making the embedding easier to predict, because the numerator and the
denominator are scored by the same fixed readout in the same space — the three groups
order as baseline 0.80-0.88 (mean 0.827), control 0.87-0.91 (mean 0.890), action
target 0.92-1.00 (mean 0.962). The control overlaps the baseline at one pair, its
0.865 falling below the baseline's 0.884, but it is separated from the action target:
its best seed, 0.907, is below the action target's worst, 0.925. So the control does
improve the scale-free prediction measure, modestly, and the action target improves it
more. The seed-0 reading, that the control does not improve prediction at all, was too
strong; what three seeds support is that the control improves it less and that the
action target's advantage over the control is seed-robust. The MSE still ranks the
control first, so the two measures still disagree on the top of the order, and the
ratio is the one that survives the regime shift described above.

**On content the three groups separate on train kits and only trend apart off them.**
The mel contains the hits, so a mel target restores onset content on train kits, but
not to the action target's level: 0.50-0.52 against 0.58-0.66, both far above the
baseline's 0.22-0.27, and all three ranges disjoint. On held-out kits the control
lands at raw-mel level (0.248-0.263 against raw mel's 0.277), clear of the baseline's
0.19-0.21 but overlapping the action target's 0.25-0.40. Two of the action target's
three seeds, 0.396 and 0.304, beat every control seed; the third, 0.253, does not. So
the held-out content gain is action-specific in direction and on the mean (0.318
against 0.254), not seed by seed.

**The control is the tightest of the three groups across seeds.** On the readout ratio
the control spans 0.04 (0.865-0.907), the baseline 0.09 (0.796-0.884) and the action
target 0.08 (0.925-1.003). The risk the seed-0 version of this document flagged — a
control seed drawn high closing the gap to the action target — did not appear: the
control's highest seed is still 0.02 below the action target's lowest.

**Effective-weight caveat.** The mel term collapsed within two epochs in every seed,
from 0.18 to 0.03, and reached 0.013-0.016 by epoch 19, while the action term stayed
near 0.3 all run. Matched at initialization is not matched throughout: for most of
training the control ran as a much weaker nudge than the action term. That is a property of the target
rather than of the weight, since a time-averaged version of the input is trivially
recoverable from tokens of that input, and rescaling the coefficient would not fix
it. Closing the general question — does any weak regularizer do this? — needs a
second control whose loss cannot collapse, such as a variance-covariance term. What
this run closes is the narrower question of whether any reconstruction target does
it.

**Against the plan's pass/fail this is still neither.** The plan said pass if the
control does not halve the normalized error and fail if it does. It more than halves
it in every seed (0.021-0.035 against 0.079-0.108), which is a literal fail, but it
does so by entering the regime where the metric stops measuring what it was chosen to
measure. The substantive reading, on three seeds, is that a weak reconstruction term
of either kind restores content to the state and improves the scale-free prediction
measure, and that the action target does both more strongly: its advantage is
seed-robust on train-kit content and on the readout ratio, and suggestive but not
separated on held-out-kit content. So the headline is narrowed rather than kept or
dropped. "A weak action-reconstruction term restores action content" becomes "a weak
reconstruction term asks the state to keep more of its input, with the action as the
best target tested".

## Caveats

The full-mask condition is out of distribution for all three models: they are trained
at 75% masking and scored here at 100%. docs/experiments.md flags this for E2 and it
applies to the full-mask readout columns too.

The E5 probe reads the same frequency-pooled tokens that both aux heads train on, so
its rise under either regularizer is close to tautological. E4 would be the
independent check; it has not been run on the control.

The item 1 readout ratio is at ceiling in the `predicted` condition for every model
scored so far, so only the full-mask column separates them. This inherits item 1's
caveat that the ratio ranks encoder content when tokens leak and prediction only when
they do not.

Kit accuracy under full masking is at ceiling for both regularized models
(0.994-0.996 for the action target, 0.996-0.998 for the control) and does not
distinguish them.
