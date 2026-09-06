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

### E3 addendum — the same geometry on AO-JEPA (2026-09-06, docs/e3/aojepa/)
Running E3 on the audio-only model (drumjepa_v1_aojepa, epoch 19) shows the kit
geometry is a product of masked JEPA training on this data, not of action
conditioning.

| test | drum-JEPA | AO-JEPA | random-init | raw mel |
|---|---|---|---|---|
| kit-vector consistency ratio, within / between | 4.6 | 4.7 | 1.8 | 3.4 |
| kit-vector transfer, all pairs | 0.731 | 0.580 | 0.112 | 0.302 |
| kit-vector transfer, both train kits | 0.943 | 0.767 | 0.146 | 0.377 |
| predictor swap, target fixed, held-out B | 0.889 | 0.942 | | |
| 14-way kit probe, linear, test | 0.923 | 0.885 | 0.908 | 0.996 |

Action conditioning leaves the geometry somewhat cleaner (transfer 0.73 vs 0.58)
but does not create it. The E3 verdict "built by training" stands; "built by
actions" would be wrong.

### E4 — done (2026-09-06, inverse model on frozen states, docs/e4/)
A 3.5M-param inverse model h reads the frozen tokens of x_t and x_{t+1} and
decodes the frame-level drumroll and hi-hat track of the transition. Trained 8
epochs on 20k transitions per representation, identical h throughout, onset
threshold tuned on validation. Onset F1 at 50 ms tolerance, test split.

| representation | train kits, macro F1 | held-out kits, macro F1 | velocity MAE (MIDI) |
|---|---|---|---|
| random-init encoder | 0.251 [0.235, 0.259] | 0.215 | 17.1 |
| raw mel, same patching | 0.250 [0.236, 0.256] | 0.221 | 17.5 |
| AO-JEPA | 0.164 [0.150, 0.170] | 0.165 | 21.4 |
| drum-JEPA | 0.118 [0.102, 0.126] | 0.119 | 23.6 |
| drum-JEPA + a_t | 0.126 | 0.129 | 22.8 |

Drum-JEPA is last and loses to both controls by a factor of two. Handing h the
previous action barely helps, so this is not groove continuation. The absolute F1s
are low for everyone because the decoder is small and the positive-class weight is
capped; the ranking is what E4 establishes.

### E5 — done (2026-09-06, content probes vs AO-JEPA, docs/e5/)
Per-250 ms-step probes on frozen, frequency-pooled state tokens, targets from the
action of the same window. Linear onset probe, macro F1 at threshold 0.5, test split.

| representation | train kits | held-out kits | 16 tokens concatenated (2k-clip subset) |
|---|---|---|---|
| raw mel | 0.583 | 0.277 | |
| random-init encoder | 0.365 | 0.320 | |
| AO-JEPA | 0.278 [0.258, 0.292] | 0.257 | 0.546 |
| drum-JEPA | 0.223 [0.206, 0.234] | 0.188 | 0.482 |
| class-prior baseline | 0.130 | 0.130 | |

Velocity, timing and pedal position are recoverable only weakly from any
representation (10-25% below the predict-the-mean baseline). Mean-pooling over
frequency hides about half the onset content; the ordering survives without it.

Verdict on Q1 across E2-E5. The state of an action-conditioned JEPA carries what
the action does not determine, and drops what it does. E2: s_t improves prediction
beyond the action, mostly through non-kit context. E3: kit identity is organized as
a consistent direction, but AO-JEPA organizes it nearly as well. E4 and E5: action
content (which drum, when, how hard) is less recoverable from drum-JEPA's state than
from an untrained encoder, and removing action conditioning (AO-JEPA) recovers part
of it. The mechanism is the objective: f receives a_{t+1} at every step, so the
state encoder is never asked to encode it and 20 epochs of training discard it.

What this means for the project. Drum-JEPA is not a better representation of drum
audio than AO-JEPA on any probe run so far; it is a differently specialized one.
The counterfactual kit lever still works (E1 kit swap 0.993, E3 swap 0.89-0.94), but
it works for AO-JEPA too. The result that is specific to action conditioning is the
negative one: action content is removed from the state. Whether that is a feature
(a cleaner "environment" latent) or a defect depends on what the state is for, and
that is the question to put to Ziyu before E6.

Caveats. One seed per model; the E4 decoder was trained for a fixed 8 epochs and
raw_mel+a_t was still improving; E5 probes use frequency-pooled tokens by default;
the AO-JEPA predictor has 3.2M rather than 4.8M parameters (no cross-attention).
