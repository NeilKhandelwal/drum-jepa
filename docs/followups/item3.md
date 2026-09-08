# Item 3 — longer baseline

Follow-up 3 of docs/followups.md, run 2026-09-07. E2 recorded the baseline still
improving at epoch 19, so the option A gain might be nothing but faster
convergence: the same world model, reached sooner. This item tests that by giving
the baseline twice the training. configs/drumjepa_v1_e40.yaml is
configs/drumjepa_v1.yaml with `epochs: 40`, `keep_epochs: [19]` and a new run
name; aux_rec stays 0, the seed stays 0, and data, masks and every other
hyperparameter are unchanged. The epoch-19 checkpoint was kept and evaluated as a
sibling run directory. It is a mid-schedule reading, not a second 20-epoch run:
the cosine schedule in scripts/train.py is built over `len(train_dl) * epochs`
steps, so the 40-epoch run is still at a high learning rate where the 20-epoch run
has annealed.

## Artifacts and commands

Copied from runs/drumjepa_v1_e40/ into docs/followups/item3/ with the prefix
`drumjepa_v1_e40_`: `train_config.json`, `train_metrics.csv`,
`e1_results.{json,md}`, `e1_win_rates.png`, `e2_results.{json,md}`,
`e5_results.{json,md}`, `e5_f1_by_class.png`, `readout_results.{json,md}`. The
same eval files from runs/drumjepa_v1_e40_at20/ carry the prefix
`drumjepa_v1_e40_at20_`; that directory has no config or metrics of its own
because it is a symlinked snapshot of the same run at epoch 19.

Training:

    .venv/bin/python scripts/train.py --config configs/drumjepa_v1_e40.yaml

The snapshot directory, as in scripts/run_followups_chain.sh: the evals read only
`config.json` and `last.pt` from a run directory, so a copy and a symlink are
enough.

    mkdir -p runs/drumjepa_v1_e40_at20
    cp runs/drumjepa_v1_e40/config.json runs/drumjepa_v1_e40_at20/config.json
    ln -sf ../drumjepa_v1_e40/epoch19.pt runs/drumjepa_v1_e40_at20/last.pt

Evals, as in scripts/run_followups_chain.sh, on each of the two run directories:

    .venv/bin/python scripts/eval_e1.py --run-dir <run-dir>
    .venv/bin/python scripts/eval_e2.py --full <run-dir> \
        --action-only runs/drumjepa_v1_actiononly
    .venv/bin/python scripts/eval_e5.py --drumjepa <run-dir> --skip aojepa
    .venv/bin/python scripts/eval_readout.py --run-dir <run-dir> --split test

## Results

Seed 0 for every row. E1 and E2 are on the validation split, E5 and the readout on
test. The baseline row is the 20-epoch drumjepa_v1 seed 0 from
docs/e2/results_validation.json, docs/e5/results.json, docs/e1/ and
runs/drumjepa_v1/readout/; the action-target row is drumjepa_v1_auxrec01 seed 0,
carried over for scale.

Prediction, MSE-based. Errors are masked MSE in each model's own teacher space;
normalized error divides the clean error by that model's squared final teacher std.

| model | epoch | teacher std | E2 clean err | E2 kit-swap err | E2 full-mask err | clean err / var | E2 win vs action-only |
|---|---|---|---|---|---|---|---|
| baseline, 20-epoch schedule | 19 | 1.081 | 0.095 | 0.115 | 0.624 | 0.082 | 0.780 |
| 40-epoch run, epoch-19 snapshot | 19 | 1.239 | 0.218 | 0.424 | 0.965 | 0.142 | 0.094 |
| 40-epoch run, end state | 39 | 1.211 | 0.097 | 0.265 | 0.667 | 0.066 | 0.776 |
| action target 0.1, 20 epochs | 19 | 1.333 | 0.067 | 1.734 | 0.164 | 0.038 | 0.931 |

E1 perturbation win rates, within model, chance 0.5.

| model | epoch | kit swap | random state | state shift |
|---|---|---|---|---|
| baseline, 20-epoch schedule | 19 | 0.993 | 0.973 | 0.594 |
| 40-epoch run, epoch-19 snapshot | 19 | 0.995 | 0.976 | 0.582 |
| 40-epoch run, end state | 39 | 0.988 | 0.946 | 0.546 |
| action target 0.1, 20 epochs | 19 | 0.997 | 0.990 | 0.633 |

Content and the item 1 readout. E5 is the linear onset probe on the test split,
macro F1 over 14 classes. Readout columns are onset F1 in the fixed
teacher-embedding readout of item 1; `full-mask / ceiling` is the scale-free
prediction measure.

| model | epoch | E5 train kits | E5 held-out | readout predicted F1 | ceiling | full-mask F1 | full-mask / ceiling | full-mask kit acc |
|---|---|---|---|---|---|---|---|---|
| baseline, 20-epoch schedule | 19 | 0.223 | 0.188 | 0.342 | 0.342 | 0.302 | 0.884 | 0.845 |
| 40-epoch run, epoch-19 snapshot | 19 | 0.223 | 0.176 | 0.342 | 0.338 | 0.290 | 0.858 | 0.925 |
| 40-epoch run, end state | 39 | 0.197 | 0.162 | 0.311 | 0.310 | 0.269 | 0.867 | 0.844 |
| action target 0.1, 20 epochs | 19 | 0.582 | 0.396 | 0.614 | 0.609 | 0.563 | 0.925 | 0.994 |

Seed ranges from the three-seed 20-epoch runs (docs/experiments.md "Seeds",
docs/followups/item1.md), for reading the single-seed 40-epoch run against:
normalized error 0.077-0.105 for the baseline and 0.026-0.038 for the action
target; readout full-mask / ceiling 0.80-0.88 and 0.92-1.00; E5 onset probe
0.22-0.27 and 0.58-0.66 on train kits, 0.19-0.21 and 0.25-0.40 held out.

Training curve (drumjepa_v1_e40_train_metrics.csv): validation state loss is 0.208
at epoch 9, 0.224 at epoch 19, 0.148 at epoch 29 and 0.101 at epoch 39. The
learning rate at epoch 19 is 3.4e-4 here against 6e-5 at the same epoch in the
20-epoch run, because the cosine horizon is the total epoch count.

## Verdict

**Passed on the plan's criterion.** The plan said pass if the epoch-40 baseline is
still clearly behind aux_rec 0.1 at epoch 20, and it is, on every measure that was
meant to carry the comparison. Its normalized error is 0.066, just below its own
20-epoch three-seed range of 0.077-0.105 and nowhere near the action target's
0.026-0.038. Its readout full-mask / ceiling ratio is 0.867, inside the baseline's
20-epoch range of 0.80-0.88 and below the action target's worst seed. Its E2 win
rate against the action-only predictor is unchanged, 0.776 against 0.780. Doubling
the schedule does not close the gap on any measure, so the option A result is not
a statement about convergence speed.

**The one number that moved is the one item 2 warned about.** Normalized error
falls from 0.082 to 0.066, which reads as a 20% improvement until the two factors
are separated: raw clean error is 0.097 at epoch 39 against 0.095 at epoch 19, so
prediction in absolute terms is flat, while the teacher's spread grew from 1.081
to 1.211 and the division by its square did the rest. This is the same failure
item 2 found when the mel control posted the lowest normalized error in the study
while doing nothing for the readout ratio: error divided by teacher variance
rewards spread, whatever produced the spread. Longer training produces it here as
surely as a regularizer did there. From this point on, quote the readout
full-mask / ceiling ratio and the E2 win rate, and treat normalized error as a
diagnostic that needs its two factors shown alongside it.

**Action content keeps falling with more training.** The E5 onset probe on the
state drops from 0.223 to 0.197 on train kits and from 0.188 to 0.162 held out,
both below the 20-epoch three-seed range of 0.22-0.27 and 0.19-0.21, and the
readout ceiling — what a frozen linear probe can read from the teacher's own
embedding of the true next window, with no prediction involved — falls with them,
0.342 to 0.310. That is the shedding mechanism of E4 and E5 continuing to act, not
saturating: the predictor receives a_{t+1}, so the state is never asked to keep
action content, and the longer training runs the less of it survives. The original
finding was measured at a single point on the schedule and could have been an
artifact of stopping at 20 epochs. It is not. Shedding is progressive, which
strengthens the finding and sharpens the case for the auxiliary term: the thing it
repairs gets worse on its own with training.

**The epoch-19 snapshot is a progress reading, not a second 20-epoch run.** It
sits mid-schedule at a learning rate of 3.4e-4, 5.6 times the 6e-5 the 20-epoch
run has annealed to by the same epoch, and its numbers show a model still in
motion: E2 clean error 0.218 against the baseline's 0.095, and an E2 win rate
against action-only of 0.094, which is a loss, not a win. Validation state loss
tells the same story, rising slightly from 0.208 at epoch 9 to 0.224 at epoch 19
before falling to 0.101 by epoch 39. Read it as a checkpoint of an unfinished run
and nothing else; the numbers are recorded here for completeness, and no
comparison in this document rests on them.

**State-side E1 win rates drift down at epoch 40.** Random state falls from 0.973
to 0.946 and state shift from 0.594 to 0.546, both small moves and both in the
direction of the state mattering less to the prediction as training proceeds. Kit
swap is at ceiling throughout (0.993 to 0.988) and says nothing. The drift is
consistent with the content result above rather than independent of it.

## Caveats

The 40-epoch run is one seed. Every comparison against a three-seed range is a
single point read against a spread.

The readout full-mask / ceiling ratio has a 0.08 spread across the baseline's
three 20-epoch seeds, so the 0.867 reading is not by itself evidence that anything
changed. It supports the negative claim — the ratio did not move out of the
baseline's range and toward the action target's — and not a positive one.

Forty epochs is one point on the schedule axis. It rules out the specific "the
baseline just needs twice as long" reading. It does not rule out a much longer
schedule behaving differently, and the E5 trend suggests that if it did, it would
be in the direction of less action content, not more.
