# Item 6 — variance-covariance control

Added after items 1-5, run 2026-09-14. Item 2 left one question open: its mel
control is a reconstruction target, so it separates "action content in the state
helps" from "any reconstruction target helps" but not from "any regularizer that
spreads the embedding helps". Its loss also collapsed within two epochs, so for most
of training it ran as a weaker nudge than the action term. This item is the control
that cannot collapse and reconstructs nothing: `aux_target: vc`
(configs/drumjepa_v1_vcrec.yaml) applies VICReg's variance and covariance terms
(Bardes et al. 2022) to the same frequency-pooled s_t step vectors the action and
mel heads read, with no head and no target. Variance: mean over dimensions of
relu(1 - std) across the B*8 pooled vectors. Covariance: sum of squared off-diagonal
entries of the d x d covariance, divided by d. Equal weights on the two terms rather
than VICReg's 25:1, because the pair is matched to one recorded aux magnitude and a
second free ratio would not be identified. Weight 0.1, seed 0, 20 epochs, everything
else as drumjepa_v1_auxrec01; seeds 1 and 2 from configs/drumjepa_v1_vcrec_s{1,2}.yaml,
identical except for the seed and run name. notes/decisions.md ("Variance-covariance
control, 2026-09-14") records the weight choice.

## Artifacts and commands

Copied from runs/drumjepa_v1_vcrec{,_s1,_s2}/ into docs/followups/item6/ under the
prefixes `drumjepa_v1_vcrec_`, `drumjepa_v1_vcrec_s1_`, `drumjepa_v1_vcrec_s2_`:
`train_config.json`, `train_metrics.csv`, `e1_results.{json,md}`,
`e2_results.{json,md}`, `e5_results.{json,md}`, `readout_results.{json,md}`, plus
`e1_win_rates.png` and `e5_f1_by_class.png` for seed 0.

All three seeds, training and evals, from one chain:

    caffeinate -i nohup bash scripts/run_vc_chain.sh > runs/vc_chain.log 2>&1 &

which runs, per seed, the same four evals as scripts/run_followups_chain2.sh (E1 on
validation, E2 against runs/drumjepa_v1_actiononly, E5 with AO-JEPA skipped, readout
on test). Chain time 4 h 48 min on the M5 Max.

## Results

Three seeds per group. Baseline, action-target and mel-control rows are the ones
recorded in docs/followups/item2.md; the vc rows come from the run directories.
Readout and E5 are on the test split, E2 and E1 on validation.

Prediction and dynamics. `clean err / var` divides the E2 clean error by the squared
final teacher std; it is retired as a cross-model claim (item 2) and shown only so
the row is complete.

| group | seed | teacher std | E2 clean err | clean err / var | E2 win vs action-only | E2 kit-swap / clean err | E2 full-mask err | E1 random state | E1 kit swap | E1 random action |
|---|---|---|---|---|---|---|---|---|---|---|
| aux_rec 0 | 0 | 1.081 | 0.095 | 0.082 | 0.780 | 1.2 | 0.624 | 0.973 | 0.993 | 0.993 |
| aux_rec 0 | 1 | 0.989 | — | 0.079 | 0.915 | 1.7 | — | 0.980 | 0.995 | — |
| aux_rec 0 | 2 | 1.039 | — | 0.108 | 0.635 | 1.3 | — | 0.979 | 0.994 | — |
| action target 0.1 | 0 | 1.333 | 0.067 | 0.038 | 0.931 | 25.8 | 0.164 | 0.990 | 0.997 | 1.000 |
| action target 0.1 | 1 | 1.391 | — | 0.026 | 0.978 | 60.4 | — | 0.991 | 0.995 | — |
| action target 0.1 | 2 | 1.374 | — | 0.033 | 0.941 | 48.5 | — | 0.990 | 0.997 | — |
| mel target 0.1 | 0 | 1.113 | 0.026 | 0.021 | 0.965 | 77.3 | 0.033 | 0.985 | 0.998 | 0.963 |
| mel target 0.1 | 1 | 1.091 | — | 0.035 | 0.928 | 48.7 | — | 0.984 | 0.998 | — |
| mel target 0.1 | 2 | 1.098 | — | 0.024 | 0.956 | 70.2 | — | 0.987 | 0.998 | — |
| vc 0.1 | 0 | 1.315 | 0.176 | 0.102 | 0.242 | 1.1 | 2.147 | 0.920 | 0.952 | 0.979 |
| vc 0.1 | 1 | 1.328 | 0.172 | 0.098 | 0.279 | 1.0 | 2.522 | 0.812 | 0.852 | 0.800 |
| vc 0.1 | 2 | 1.392 | 0.223 | 0.115 | 0.118 | 1.1 | 2.479 | 0.892 | 0.940 | 0.884 |

The action-only predictor's clean error on the same transitions is 0.144; every vc
seed's own clean error is above it (0.172-0.223), which is why the E2 win rate is
below 0.5 in all three. E1 state shift for vc is 0.508-0.548 against 0.594-0.654
for the baseline.

Content and the readout ratio. E5 is the linear onset probe, macro F1 over 14
classes; the readout columns are item 1's fixed teacher-embedding readout, and
`full-mask / ceiling` is the scale-free prediction measure. `ceiling kit acc` is the
readout's 14-way kit probe on the teacher's encoding of the real next window.

| group | seed | E5 train kits | E5 held-out kits | readout ceiling F1 | full-mask F1 | full-mask / ceiling | ceiling kit acc | full-mask kit acc |
|---|---|---|---|---|---|---|---|---|
| aux_rec 0 | 0 | 0.223 | 0.188 | 0.342 | 0.302 | 0.884 | 0.984 | 0.845 |
| aux_rec 0 | 1 | 0.265 | 0.208 | 0.381 | 0.306 | 0.802 | 0.990 | 0.922 |
| aux_rec 0 | 2 | 0.263 | 0.209 | 0.386 | 0.308 | 0.796 | 0.989 | 0.928 |
| action target 0.1 | 0 | 0.582 | 0.396 | 0.609 | 0.563 | 0.925 | 0.995 | 0.994 |
| action target 0.1 | 1 | 0.618 | 0.253 | 0.648 | 0.621 | 0.958 | 0.996 | 0.996 |
| action target 0.1 | 2 | 0.664 | 0.304 | 0.683 | 0.686 | 1.003 | 0.995 | 0.996 |
| mel target 0.1 | 0 | 0.522 | 0.250 | 0.571 | 0.494 | 0.865 | 0.996 | 0.996 |
| mel target 0.1 | 1 | 0.500 | 0.263 | 0.552 | 0.501 | 0.907 | 0.995 | 0.996 |
| mel target 0.1 | 2 | 0.505 | 0.248 | 0.551 | 0.494 | 0.897 | 0.996 | 0.998 |
| vc 0.1 | 0 | 0.129 | 0.128 | 0.245 | 0.217 | 0.886 | 0.939 | 0.236 |
| vc 0.1 | 1 | 0.166 | 0.170 | 0.270 | 0.203 | 0.752 | 0.801 | 0.220 |
| vc 0.1 | 2 | 0.148 | 0.154 | 0.256 | 0.202 | 0.789 | 0.711 | 0.268 |

Group means of the readout ratio: baseline 0.827, action target 0.962, mel control
0.890, vc 0.809. E5 held-out: 0.202, 0.318, 0.254, 0.151. E5 controls, unchanged
from docs/e5/: random-init encoder 0.365 train kits / 0.320 held-out, raw mel
0.583 / 0.277, class-prior baseline 0.130.

Training curves (drumjepa_v1_vcrec*_train_metrics.csv). The vc term is 1.28 at step
50, 0.69 averaged over epoch 0, 0.34 at epoch 2 and 0.27 at epoch 19, the same in
all three seeds. The variance half is satisfied by epoch 2 (teacher std 0.59 at
epoch 0, 1.0 at epoch 2, 1.32-1.39 at epoch 19), so the 0.27 that remains for the
last 17 epochs is the covariance half, which never gets close to zero. Validation
state loss at epoch 19 is 0.170-0.227 against 0.076-0.118 for the baseline,
0.048-0.069 for the action target and 0.027-0.040 for the mel control.

## Verdict

**The vc control is worse than the unregularized baseline on every measure.** The
readout ratio, 0.75-0.89 (mean 0.809), overlaps the baseline's 0.80-0.88 and is below
both reconstruction targets, whose worst seeds (mel 0.865, action 0.925) are above
every vc seed except one (0.886 against mel's 0.865). Content is not restored but
removed: E5 onset F1 on train kits is 0.13-0.17 against the baseline's 0.22-0.27, and
the readout ceiling, which scores the teacher's encoding of the real window, falls
from 0.34-0.39 to 0.25-0.27. Prediction gets worse in absolute terms: the model's
own clean error rises above the action-only predictor's, so the E2 win rate is
0.12-0.28 where every other group is above 0.6, and the full-mask error is 3.4-4.0
times the baseline's. E1 sensitivity drops on every perturbation, kit swap 0.85-0.95
against 0.993-0.998.

**Kit identity leaves the state.** The readout's kit probe on the teacher's own
encoding of the real window falls to 0.71-0.94 from 0.98-1.00 for every other
model, and on the predicted state under a full mask to 0.22-0.27 from 0.85-0.93
(baseline) and 0.99+ (both reconstruction targets). Kit-swapped error is within 3-12%
of clean error, against 20-70% for the baseline and 26-77x for the reconstruction
targets. The covariance penalty decorrelates the pooled step dimensions, and the
mean-pooled clip embedding, which is where the kit lives (E3), loses its linear
separability with them.

**Reading.** The two reconstruction targets and the vc term were matched at one
weight, and the vc term stayed at 0.27 for 17 epochs where the mel term collapsed to
0.015 and the action term sat near 0.3. So the vc run is, if anything, the control
whose effective weight is closest to the action target's throughout, and it goes the
other way on every axis. That separates "any reconstruction target" from "any
regularizer": spreading the same pooled vectors without asking them to carry
anything does not restore content and does not help prediction. The narrowed
headline from item 2 stands and gains its missing clause: a weak reconstruction term
keeps input content in the state and helps the scale-free prediction measure, the
action target is the best target tested, and a target-free spreading term of matched
magnitude hurts both.

**The mechanism story this supports.** Option A's gain comes from what the aux term
asks the state to keep, not from the extra gradient or the extra spread. A term that
enlarges teacher std as much as the action target does (1.32-1.39 against 1.33-1.39)
without a content target moves the state away from the JEPA target rather than
toward it: decorrelating the step vectors that the predictor has to match makes the
target harder to predict, which is the opposite of the regime shift item 2 found for
the reconstruction targets.

## Caveats

One weight only. 0.1 was matched to the action head's magnitude at initialization
(notes/decisions.md); a smaller coefficient might be neutral rather than harmful, and
this run does not say where that boundary is. Nothing here suggests a smaller vc
weight would help, only that it would hurt less.

Equal weighting of the variance and covariance halves is a choice, not VICReg's
setting. Since the variance half is satisfied by epoch 2, the run is in practice a
covariance-only regularizer for 90% of training, so the finding is about
decorrelation of the pooled steps specifically.

The term is applied to the 256-d frequency-pooled step vectors, not to the full
token grid, so that it reads exactly what the action and mel heads read. A VICReg
term on the un-pooled tokens is a different intervention and was not run.

The E5 probe and the readout share the pooled tokens the aux terms act on, as in
items 1 and 2. E4 (an independent decoder on un-pooled tokens) has not been run on
the vc control; given that every pooled measure fell, it was not needed to read the
result.
