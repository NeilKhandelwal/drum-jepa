# Item 5 — prediction on unseen kits for the aux_rec 0.1 model

Follow-up 5 of docs/followups.md, run 2026-09-07. Content on held-out kits was
already measured for the 0.1 model (E5, E4). Prediction on held-out kits was not.
E3's predictor swap and E1's perturbation win rates both have a held-out-kit form,
and neither had been run on drumjepa_v1_auxrec01. E3 was extended the same day to
seeds 1 and 2 of both weights, because its held-out swap gap was the one number in
this item that a single seed could not support.

Two evals. E3 is unchanged and reports its held-out-B swap alongside the train-B
one. E1 gained a `--kits` flag: it selects the kit subset the transitions are drawn
from, and every perturbation source, kit swap included, is drawn from that same
subset, so `--kits heldout` swaps one held-out kit's state for another's rather than
reaching back into a train kit. Held-out output goes to `<run-dir>/e1_heldout`, so
the train-kit E1 results in each run directory are untouched.

Commands:

    scripts/eval_e3.py --run-dir runs/drumjepa_v1_auxrec01
    scripts/eval_e1.py --run-dir runs/drumjepa_v1_auxrec01 --kits heldout
    scripts/eval_e1.py --run-dir runs/drumjepa_v1 --kits heldout

Seed 0's baseline E3 numbers are read from docs/e3/results.json, not rerun. E3 was
then added for seeds 1 and 2 of both weights, with the same arguments the seed 0
runs used (`--max-train 40000 --max-swaps 4000 --seed 0 --bs 256`, all defaults):

    scripts/eval_e3.py --run-dir runs/drumjepa_v1_s1
    scripts/eval_e3.py --run-dir runs/drumjepa_v1_s2
    scripts/eval_e3.py --run-dir runs/drumjepa_v1_auxrec01_s1
    scripts/eval_e3.py --run-dir runs/drumjepa_v1_auxrec01_s2

Artifacts: docs/followups/item5/.

## Table 1 — E3 kit geometry and predictor swap

Seed 0 of each weight. Frozen teacher encoder, mean-pooled 256-d clip embedding,
2034 swap windows, 1000 bootstrap resamples. Chance is 0.5 for the swaps.

| test | drum-JEPA (aux_rec 0) | aux_rec 0.1 |
|---|---|---|
| predictor swap, target fixed, B a train kit | 0.992 [0.986, 0.997] | 0.995 [0.989, 0.999] |
| predictor swap, target fixed, B a held-out kit | 0.889 [0.871, 0.905] | 0.955 [0.943, 0.967] |
| predictor swap, all B | 0.929 [0.917, 0.940] | 0.971 [0.963, 0.979] |
| kit-vector transfer, all 182 ordered pairs | 0.731 | 0.761 |
| kit-vector transfer, both kits in train | 0.943 | 0.954 |
| kit-vector consistency, within / between cosine | 0.316 / 0.068 | 0.637 / 0.129 |

## Table 2 — E3 across three seeds

Same eval, same arguments, on the three training seeds of each weight. Brackets are
the 1000-resample bootstrap interval on the held-out-B swap. Chance is 0.5 for the
swaps.

| weight | seed | swap, train B | swap, held-out B | swap, all B | transfer, all pairs | transfer, both train |
|---|---|---|---|---|---|---|
| aux_rec 0 | 0 | 0.992 | 0.889 [0.871, 0.905] | 0.929 | 0.731 | 0.943 |
| aux_rec 0 | 1 | 0.994 | 0.886 [0.869, 0.903] | 0.928 | 0.745 | 0.968 |
| aux_rec 0 | 2 | 0.996 | 0.893 [0.873, 0.913] | 0.933 | 0.735 | 0.943 |
| aux_rec 0 | mean | 0.994 | 0.889 | 0.930 | 0.737 | 0.951 |
| aux_rec 0.1 | 0 | 0.995 | 0.955 [0.943, 0.967] | 0.971 | 0.761 | 0.954 |
| aux_rec 0.1 | 1 | 0.995 | 0.917 [0.902, 0.930] | 0.947 | 0.752 | 0.889 |
| aux_rec 0.1 | 2 | 0.994 | 0.945 [0.930, 0.958] | 0.964 | 0.761 | 0.923 |
| aux_rec 0.1 | mean | 0.994 | 0.939 | 0.960 | 0.758 | 0.922 |

## Table 3 — E1 win rates, train kits against held-out kits

Validation split, K=4 masks per transition, cluster bootstrap over sequences.
Train-kit columns are the recorded runs: docs/e1 for the baseline,
docs/optionA/sweep for aux_rec 0.1. Held-out columns are the new runs, 16272
transitions each against 12204 for the train-kit runs.

| perturbation | aux_rec 0, train | aux_rec 0, held-out | aux_rec 0.1, train | aux_rec 0.1, held-out |
|---|---|---|---|---|
| state time-shift | 0.594 [0.566, 0.620] | 0.523 [0.504, 0.542] | 0.633 [0.586, 0.680] | 0.526 [0.513, 0.539] |
| random state | 0.973 [0.968, 0.977] | 0.829 [0.813, 0.844] | 0.990 [0.987, 0.992] | 0.836 [0.824, 0.847] |
| action time-shift | 0.979 [0.966, 0.989] | 0.934 [0.911, 0.952] | 0.997 [0.994, 1.000] | 0.986 [0.975, 0.995] |
| random action | 0.993 [0.990, 0.995] | 0.941 [0.932, 0.948] | 1.000 [1.000, 1.000] | 0.991 [0.987, 0.995] |
| kit swap | 0.993 [0.987, 0.997] | 0.839 [0.820, 0.856] | 0.997 [0.995, 0.999] | 0.839 [0.824, 0.852] |

## Verdict

Passed, and the held-out swap gap separates across seeds. The bar was a held-out-B
predictor swap win rate at or above drumjepa_v1's 0.889. Over three seeds the
baseline spans 0.886 to 0.893 and the 0.1 model spans 0.917 to 0.955, so the two
sets do not overlap: every 0.1 seed beats every baseline seed, means 0.939 against
0.889. Follow the note in docs/followups.md and call this "no overlap across three
seeds," not "significant"; 3 against 3 cannot reach significance by a rank test. The
margin is also thinner than seed 0 alone suggested. The closest pair is baseline
seed 2 at 0.893 against 0.1 seed 1 at 0.917, a 0.024 gap whose bootstrap intervals
([0.873, 0.913] and [0.902, 0.930]) do overlap, and the 0.1 model's own spread
across seeds (0.038) is more than half its mean advantage. Train-B is at ceiling for
both (0.994 against 0.994 on the mean) and does not separate on any seed, so
held-out B is where the difference lives: the regularized encoder's kit information
is more useful to the predictor on kits it never trained on, not just on kits it
did. The all-B swap follows held-out B, 0.930 against 0.960, and also separates
(0.928-0.933 against 0.947-0.971).

Kit-vector transfer over all pairs moves the same way but only slightly, 0.737
against 0.758 on the mean, and it does separate (0.731-0.745 against 0.752-0.761).
Transfer between train kits does not, and on three seeds it reverses: the baseline
means 0.951 (0.943-0.968) against 0.922 (0.889-0.954) for the 0.1 model, so the seed
0 reading of 0.943 to 0.954 was the wrong sign of a difference that is inside seed
noise. Read the two together as: the geometry itself is barely changed, and between
train kits it may be slightly worse; what changed is how well the predictor exploits
it on unseen kits. The consistency ratio tells a similar story from a different
angle: at seed 0, within-pair cosine doubles from 0.32 to 0.64 while between-pair
only rises from 0.07 to 0.13, so displacements between the same performance on two
kits point the same way far more sharply at weight 0.1. That measurement is now
available on all six runs in docs/followups/item5/ but was not tabulated here.

E1 on held-out kits is the part that does not flatter the 0.1 model. Both models
lose a lot when the transitions come from unseen kits: kit swap falls from 0.99 to
0.84 for both, and random state from 0.97-0.99 to 0.83-0.84. On these two
state-side perturbations the two models are indistinguishable on held-out kits
(0.839 against 0.839; 0.829 against 0.836), even though the 0.1 model separates
cleanly from the baseline on train kits. The action-side perturbations do separate
on held-out kits: action time-shift 0.986 against 0.934 and random action 0.991
against 0.941, both non-overlapping.

Read together: at weight 0.1 the world model tracks its actions better everywhere,
including on unseen kits, and it uses the state's kit content better in the E3
swap, but its raw sensitivity to a corrupted state on unseen kits is no better than
the baseline's. The held-out swap claim in the headline stands, on three seeds. A broader claim
that the 0.1 model generalizes to unseen kits across the board does not.

## Caveats

E3 now has three seeds per weight, so the held-out swap no longer rests on seed 0.
E1 on held-out kits still does: Table 3 is one seed per model, and the seeds work in
docs/optionA/seeds/ covers E1 on train kits and E2, not E1 on held-out kits. The
action-side separations in Table 3 are therefore single-seed claims, and given that
item 4 found a 0.084 spread across seeds on held-out E4 F1, they carry the weakness
that E3 has now shed. Two more seeds of eval_e1 --kits heldout would settle it and
need no training.

E1's held-out and train-kit runs differ in sample size (16272 against 12204) because
there are eight held-out kits and six train kits. The comparison across those two
columns is between different transition sets, not a paired one.

The E3 swap and the E1 kit swap disagree about held-out kits, 0.955 against 0.839
for the same model, both at seed 0. They are different measurements: the E3 swap fixes the target
at kit B's next clip and varies only which kit's state the predictor gets, while E1
swaps the state and keeps the original target. The E3 form is easier and the E1 form
is the stricter test of whether the state is being used. Quote both, not one.

Held-out kits remain packed near Ele-Drum in the encoder's geometry, as the original
E3 caveat recorded. Nothing in these runs changes that; the transfer number between
two held-out kits is 0.547 at weight 0.1 against 0.476 at weight 0, still far below
the 0.95 between train kits.
