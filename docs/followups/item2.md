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
masks, recipe, seed 0, 20 epochs — matches drumjepa_v1_auxrec01.

## Artifacts and commands

Copied from runs/drumjepa_v1_melrec/ into docs/followups/item2/, prefix
`drumjepa_v1_melrec_`: `train_config.json`, `train_metrics.csv`,
`e1_results.{json,md}`, `e1_win_rates.png`, `e2_results.{json,md}`,
`e5_results.{json,md}`, `e5_f1_by_class.png`, `readout_results.{json,md}`.

Training:

    .venv/bin/python scripts/train.py --config configs/drumjepa_v1_melrec.yaml

Evals, as in scripts/run_followups_chain.sh:

    .venv/bin/python scripts/eval_e1.py --run-dir runs/drumjepa_v1_melrec
    .venv/bin/python scripts/eval_e2.py --full runs/drumjepa_v1_melrec \
        --action-only runs/drumjepa_v1_actiononly
    .venv/bin/python scripts/eval_e5.py --drumjepa runs/drumjepa_v1_melrec --skip aojepa
    .venv/bin/python scripts/eval_readout.py --run-dir runs/drumjepa_v1_melrec --split test

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

Seed ranges from the three-seed runs (docs/followups/item1.md, docs/experiments.md
"Seeds"), for reading the single-seed control against: full-mask / ceiling is 0.88,
0.80, 0.80 for aux_rec 0 and 0.92, 0.96, 1.00 for the action target; E5 held-out is
0.19-0.21 for aux_rec 0 and 0.25-0.40 for the action target.

Training curves (drumjepa_v1_melrec_train_metrics.csv, and the auxrec01 metrics for
comparison): the mel aux loss falls from 0.18 at epoch 0 to 0.03 at epoch 2 and 0.013
at epoch 19. The action aux loss stays near 0.3 for the whole run, 0.57 at epoch 0 and
0.31 at epoch 19. Validation state loss at epoch 19 is 0.098 for the baseline, 0.069
for the action target and 0.029 for the control.

## Verdict

**On the MSE metrics the control wins.** It has the lowest clean error (0.026 against
0.067 and 0.095), the lowest normalized error (0.021 against 0.038 and 0.082) and the
lowest full-mask error (0.033 against 0.164 and 0.624). Read only through the metrics
the option A sweep used, the control refutes the action-specific claim outright: a
regularizer with no action content in its target improves prediction more than the
action term does.

**Both regularized models moved into a different regime, and the control is what
exposed it.** Kit-swapped error is 15 to 20 times the clean error for both — 1.734
against 0.067 for the action target, 2.024 against 0.026 for the control — where the
baseline's is 1.2 times (0.115 against 0.095). At the same time the full-mask error
collapses toward the clean error, from 6.6 times it for the baseline to 2.4 times for
the action target and 1.3 times for the control. Both numbers say the same thing: the
predictor leans much harder on s_t, and the target has become close to a function of
s_t plus the action. Lower MSE in this regime means the target is easier to predict,
not that the world model is better. This shift was present in the action-target run
from the start and was not noticed until the control was run against it.

**The scale-free readout disagrees with the MSE.** On the full-mask/ceiling ratio from
item 1 — the one prediction measure here that cannot be improved by making the
embedding easier to predict, because the numerator and the denominator are scored by
the same fixed readout in the same space — the control's 0.865 sits inside the
baseline's three-seed range of 0.80-0.88, while the action target's three seeds
(0.92, 0.96, 1.00) sit above that range with no overlap. So on MSE the control looks
best and the action target second; on the ratio the control does not improve
prediction over the baseline at all and the action target does. The two measures
point in opposite directions, and the ratio is the one that survives the regime shift
described above.

**On content the control is action-specific only off the training kits.** The mel
contains the hits, so a mel target restores onset content on train kits nearly to the
action target's level: 0.522 against 0.582, both far above the baseline's 0.223. On
held-out kits the control lands at raw-mel level (0.250 against raw mel's 0.277) and
at the bottom of the action target's seed range (0.25-0.40, mean 0.318). The
action-specific gain in content is on unseen kits, which is also where the raw-mel
control collapses.

**The control is one seed.** Seeds 1 and 2 are queued in
scripts/run_followups_chain2.sh. The readout-ratio conclusion above is provisional
until they land: the baseline's own spread on that ratio is 0.08, and the control at
0.865 sits 0.06 below the action target's worst seed, so a control seed drawn high
would close the gap.

**Effective-weight caveat.** The mel term collapsed within two epochs, from 0.18 to
0.03, and reached 0.013 by epoch 19, while the action term stayed near 0.3 all run.
Matched at initialization is not matched throughout: for most of training the control
ran as a much weaker nudge than the action term. That is a property of the target
rather than of the weight, since a time-averaged version of the input is trivially
recoverable from tokens of that input, and rescaling the coefficient would not fix
it. Closing the general question — does any weak regularizer do this? — needs a
second control whose loss cannot collapse, such as a variance-covariance term. What
this run closes is the narrower question of whether any reconstruction target does
it.

**Against the plan's pass/fail this is neither.** The plan said pass if the control
does not halve the normalized error and fail if it does. It more than halves it
(0.021 against 0.082), which is a literal fail, but it does so by entering the regime
where the metric stops measuring what it was chosen to measure. The prediction half
of the headline survives on the readout ratio and dies on the MSE. The content half
survives on held-out kits and is shared with the control on train kits. The honest
statement of the result is that the action target is not distinguishable from a
generic reconstruction regularizer on MSE-based prediction or on train-kit content,
and is distinguishable on the readout ratio and on held-out-kit content.

## Caveats

One seed, as above. Every number in the tables is seed 0, and only the aux_rec 0 and
0.1 rows have three-seed ranges behind them.

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

Kit accuracy under full masking is at ceiling for both regularized models (0.994 and
0.996) and does not distinguish them.
