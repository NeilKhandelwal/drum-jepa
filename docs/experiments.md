# Experiment plan

Gates are sequential. Don't start a block until the previous gate passes.

## Q1 — What does the state carry when the action nearly determines the audio?
Prediction: an action-only predictor f(a_{t+1}) matches the full model f(s_t, a_{t+1})
on prediction loss, except on kit-swapped inputs where the full model wins.

## Q2 — Is kit identity recovered as a latent?
Prediction: linear probe recovers kit from s_t; counterfactual swap (same action,
s_t from kit B) moves the prediction toward kit B; held-out kits embed sensibly.

## Secondary — velocity vs choke (2x2 modulate/inject × continuous/discrete)
Only if aftertouch survived (see docs/inventory.md). Velocity MAE from the
amortized inverse; prediction-loss sensitivity to choke-vs-no-choke on segments
with a ringing open hi-hat.

## Blocks
- E1 dynamics sanity: perturbation win rates (state/action time-shift, random,
  kit swap). Gate: win rates clearly above 0.5.
- E2 state ablation: full vs action-only, on held-out sequences, kit-swapped
  inputs, post-ring-out segments.
- E3 kit latent: linear + MLP probes, counterfactual geometry, held-out kits.
- E4 inverse recovery: onset/velocity F1 and hi-hat estimate via amortized inverse.
- E5 content probes: linear vs MLP, drum-JEPA vs AO-JEPA.
- E6 SIGReg swap; characterize failure mode if any.

## Results

### E1 — passed (2026-09-05, run drumjepa_v1, epoch 19, validation split, 6 train kits)
Win rate = fraction of transitions where the clean prediction error is lower than
the error with one input corrupted; K=4 fixed multi-block masks per transition;
95% CI from a cluster bootstrap over sequences. Full table, per-kit kit-swap
breakdown, figure, training curve and run config: docs/e1/.

| perturbation | win rate [95% CI] |
|---|---|
| state time-shift (adjacent 2 s, same kit) | 0.594 [0.566, 0.620] |
| random state | 0.973 [0.968, 0.977] |
| action time-shift | 0.979 [0.966, 0.989] |
| random action | 0.993 [0.990, 0.995] |
| kit swap | 0.993 [0.987, 0.997] |

Kit swap, the perturbation the project rests on, is above 0.98 on every train
kit. State time-shift is the weakest by design: the adjacent 2 s of the same
recording is a plausible state, and the win rate is above 0.5 but not by much.
Training: 20 epochs, 12,400 steps, ~3.5 min/epoch on M5 Max; final train loss_s
0.094, validation loss_s 0.098; embedding std rose from 0.62 to 1.09 and held (no
collapse). Gate passed; E2 is unblocked.

### E2 — done (2026-09-05, full drumjepa_v1 vs action-only drumjepa_v1_actiononly, both epoch 19)
The action-only baseline is the same model with `use_state: false`: f gets no s_t
tokens, everything else (data, masks, recipe, 20 epochs) identical. Errors are per
transition under the same K=4 masks as E1. Win rate is P[err full < err action-only];
log-ratio is mean log(err action-only / err full), positive = full better. Full
tables, figures, and the baseline's training curve: docs/e2/.

| condition | split | n | err full | err action-only | win rate [95% CI] | log-ratio |
|---|---|---|---|---|---|---|
| clean | validation | 12204 | 0.095 | 0.144 | 0.780 [0.756, 0.802] | +0.32 |
| clean | test | 11760 | 0.092 | 0.139 | 0.769 [0.734, 0.798] | +0.31 |
| kit-swapped s_t | validation | 12204 | 0.116 | 0.144 | 0.615 [0.583, 0.643] | +0.13 |
| kit-swapped s_t | test | 11760 | 0.110 | 0.139 | 0.610 [0.566, 0.646] | +0.13 |
| full mask | validation | 12204 | 0.624 | 1.400 | 1.000 | +0.84 |
| full mask | test | 11760 | 0.654 | 1.438 | 1.000 | +0.81 |
| clean, ring-out (<=2 onsets) | validation | 42 | 0.097 | 0.151 | 0.595 [0.467, 0.690] | +0.21 [-0.03, +0.35] |
| clean, ring-out (<=2 onsets) | test | 36 | 0.165 | 0.288 | 0.667 [0.333, 0.733] | +0.46 [-0.39, +0.63] |

Verdict on Q1. The prediction "action-only matches the full model except on
kit-swapped inputs" does not hold. On ordinary held-out inputs the full model's
error is about 33% lower and it wins 78% of transitions, on both splits. The state
carries predictive information beyond the action.

How much of that is kit identity? Giving the full model a wrong-kit s_t raises its
error from 0.095 to 0.116, so kit identity is worth about 0.02. The remaining gap
to the action-only model (0.116 vs 0.144) is state information that survives a kit
swap: what is ringing, and the timing and dynamics context of the performance.

Under a full mask (no visible s_{t+1} tokens, so kit identity can only come from s_t)
the full model wins every transition, as predicted. Both models were trained at
75% masking, so this condition is out of distribution for both; the comparison is
symmetric but the absolute errors are not meaningful.

Ring-out windows are too rare to conclude anything: 42 and 36 transitions, and
the CIs span zero. On validation the kit-swapped ring-out row is the only place the
action-only model is nominally better, which is consistent with decaying sound
being the case where a wrong kit hurts most, but n is too small to claim it.

Caveats. The baseline's validation loss was still falling at epoch 19 (0.185 at
epoch 14, 0.147 at 19); the full model's fell over the same span (0.131 to 0.098),
so the gap is not an artifact of one run converging earlier, but a longer
schedule would tighten the estimate. The 25% visible s_{t+1} tokens leak kit
identity to both models at 75% masking, which makes the clean-condition gap a
lower bound on what s_t contributes.

### E3 — done (2026-09-06, drumjepa_v1 epoch 19, frozen teacher encoder, mean-pooled 256-d clip embedding)
Every number is bounded by two controls run through the identical pipeline: a
random-init encoder of the same architecture, and the raw mean log-mel of the clip.
Full tables and figures: docs/e3/.

| test | teacher | random-init | raw mel | chance |
|---|---|---|---|---|
| 14-way kit probe, linear, test acc | 0.923 | 0.908 | 0.996 | 0.071 |
| kit-vector consistency, within / between pair cosine | 0.316 / 0.068 | 0.353 / 0.199 | 0.500 / 0.149 |  |
| kit-vector transfer accuracy (all 182 ordered pairs) | 0.731 | 0.112 | 0.302 | ~0.07 |
| predictor swap, target fixed, B train kit | 0.992 [0.986, 0.997] | | | 0.5 |
| predictor swap, target fixed, B held-out kit | 0.889 [0.871, 0.905] | | | 0.5 |

Verdict on Q2, in three parts.

Readable, but not because of training. A linear probe reads the kit from s_t at
0.92 on 14 kits, but the raw spectrum gives 0.996 and the untrained encoder 0.91.
The encoder does not make kit identity more linearly accessible than the input
already was; if anything it discards some. The probe half of Q2 does not count.

Structured, and that is built by training. The displacement e_B - e_A between the
same performance on two kits points the same way across performances 4.6x more
than between unrelated kit pairs, against 1.8x for the untrained encoder and 3.4x
for raw mel. A kit vector estimated on half the windows moves a clip to the right
kit's centroid on the other half 73% of the time (94% between train kits, 48%
between two held-out kits), against 11% and 30% for the controls. This is the
counterfactual-swap geometry Q2 predicted, and neither control has it.

Used by the predictor, for unseen kits too. With the target fixed as kit B's next
clip, handing the predictor B's state instead of a train kit's state wins 99% of
windows when B is a train kit and 89% when B was never trained on.

Held-out kits land in plausible places: 5 of 8 map to the train kit the kit-split
annotations predicted, and every held-out kit is separable (lowest 14-way recall
0.81). The three misses (Bigga Bop, Heavy Metal, Raw Dnb) are absorbed by Ele-Drum
under the 6-way probe while being separable at 0.87-0.91 in the 14-way one, so the
6-way probe simply has no column that fits them.

Caveat. The PCA of kit centroids shows the trained encoder spreading the six train
kits wide and packing five of eight held-out kits into one cluster near Ele-Drum;
the encoder's geometry is organized around the kits it trained on. The held-out
transfer number (48%) says the same thing.
