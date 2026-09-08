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

### Option A — action reconstruction regularizer (2026-09-06, drumjepa_v1_auxrec, docs/optionA/)
Same as drumjepa_v1 plus a BCE term that reconstructs the current window's
drumroll from the frequency-pooled state tokens, weight 1.0. Tests whether the
state can keep action content without losing its prediction advantage.

| test | drum-JEPA | option A | controls |
|---|---|---|---|
| E5 onset probe, linear, test, train kits | 0.223 | 0.625 [0.599, 0.646] | random 0.365, raw mel 0.583 |
| E5 onset probe, linear, test, held-out kits | 0.188 | 0.484 [0.450, 0.498] | random 0.320, raw mel 0.277 |
| E4 inverse model, onset F1, test, train kits | 0.118 | 0.464 [0.444, 0.481] | random 0.251, raw mel 0.250 |
| E1 kit swap win rate | 0.993 | 0.935 | |
| E1 random state win rate | 0.973 | 0.893 | |
| E1 random action win rate | 0.993 | 0.996 | |
| state prediction error / teacher variance | 0.080 | 0.210 | action-only 0.111 |

Action content came back, and then some. The regularized state is the first
representation to beat both controls on the content probes, on train kits and,
by a wider margin, on held-out kits, where raw mel collapses (0.28) and option A
holds 0.48. The inverse model nearly quadruples its onset F1.

Prediction paid for it. Raw MSE is not comparable across models whose teachers
have different spread (option A's embedding std is 1.89 vs 1.09), so the fair
comparison is error divided by teacher variance: 0.21 against 0.08 for drum-JEPA
and 0.11 for the action-only baseline. The E2 advantage over action-only is gone
(win rate 0.005). E1 confirms the same shift scale-free: the model still tracks
its actions (0.99) but leans on its state less (kit swap 0.99 to 0.94, random
state 0.97 to 0.89).

Verdict. At weight 1.0 there is a real tradeoff, not a free fix: the state can be
made to carry action content, but the predictor then relies on it less and
predicts worse. This is consistent with the E4/E5 mechanism read in reverse. The
open question is the shape of the tradeoff curve; a sweep over the weight (0.1,
0.3) with the normalized error and the E1 win rates as the prediction metrics is
one run each. The weight-1.0 point also says something on its own: with the
reconstruction term this strong, the reconstruction loss (0.06) dominates the
state loss's gradient early in training, so the encoder is shaped by the
regularizer first and the prediction task second.

Protocol note. E2's cross-model error comparison silently assumed comparable
teacher scales, which held for the original pair (std 1.09 vs 1.14) and does not
hold here. Future cross-model comparisons should report error / teacher variance
or win rates only.

### Option A sweep — aux_rec 0.1 and 0.3 (2026-09-06, docs/optionA/sweep/)
Same recipe as drumjepa_v1 with the action-reconstruction term at four weights.
Prediction is compared scale-free: error divided by the teacher's mean per-dim
variance, and the E1 within-model win rates. E5 is the linear onset probe on the
test split. One seed per point.

| aux_rec | teacher std | state error / teacher var | E2 win vs action-only | E1 random state | E1 kit swap | E5 onset, train kits | E5 onset, held-out kits |
|---|---|---|---|---|---|---|---|
| 0 (drumjepa_v1) | 1.09 | 0.081 | 0.77 | 0.973 | 0.993 | 0.223 | 0.188 |
| 0.1 | 1.33 | 0.038 | 0.93 | 0.990 | 0.997 | 0.582 | 0.396 |
| 0.3 | 1.53 | 0.191 | 0.01 | 0.993 | 0.997 | 0.663 | 0.494 |
| 1.0 | 1.89 | 0.210 | 0.005 | 0.893 | 0.935 | 0.625 | 0.484 |
| controls | | action-only 0.111 | | | | random 0.365, raw mel 0.583 | random 0.320, raw mel 0.277 |

Verdict. Weight 0.1 is the point that matters. It restores action content to the
level of the raw spectrogram on train kits (0.58) and above every control on
held-out kits (0.40), and at the same time halves the normalized prediction
error relative to the unregularized model (0.038 vs 0.081) and raises the E1
state-side win rates. On this evidence a small reconstruction term is not a
tradeoff at all: it is a better drum-JEPA on every measure run so far.

Between 0.1 and 0.3 the prediction error jumps five-fold while the E1 win rates
stay at their ceiling, so the predictor still uses its state well but the target
has become harder to hit: the encoder is spreading variance along action-content
directions that the masked-token predictor cannot reproduce exactly. Content
keeps rising to 0.66 at 0.3 and then falls back at 1.0 as the regularizer starts
to dominate the encoder.

What is not established. Whether the 0.1 improvement in prediction is real or a
seed effect; the step between 0.1 and 0.3 is one run each, so its sharpness is
unknown; and E4 was not rerun on the sweep. The next spend is three seeds at 0
and 0.1, which settles the headline, and one point at 0.2 if the curve's shape
matters.

Reading across the whole project: the action-conditioned JEPA sheds action
content because nothing asks the state to keep it (E4, E5), and a weak
reconstruction term puts it back at no measured cost to the world model. That is
the method result, and it is one sentence.

### Seeds — aux_rec 0 vs 0.1, three seeds each (2026-09-07, docs/optionA/seeds/)
Seeds 0, 1, 2 at each weight, identical recipe otherwise. Per-seed values, then
the range. Normalized error = E2 clean error / squared final teacher std.

| metric | aux_rec 0, seeds 0 / 1 / 2 | aux_rec 0.1, seeds 0 / 1 / 2 | separated? |
|---|---|---|---|
| normalized state error | 0.077 / 0.077 / 0.105 | 0.038 / 0.026 / 0.033 | yes, max 0.038 < min 0.077 |
| E2 win rate vs action-only | 0.77 / 0.92 / 0.64 | 0.93 / 0.98 / 0.94 | yes |
| E1 random-state win rate | 0.973 / 0.980 / 0.979 | 0.990 / 0.991 / 0.990 | yes |
| E1 kit-swap win rate | 0.993 / 0.995 / 0.994 | 0.997 / 0.995 / 0.997 | overlap |
| E5 onset probe, train kits | 0.223 / 0.265 / 0.263 | 0.582 / 0.618 / 0.664 | yes, min 0.58 > max 0.27 |
| E5 onset probe, held-out kits | 0.188 / 0.208 / 0.209 | 0.396 / 0.253 / 0.304 | yes, min 0.25 > max 0.21 |

Mean over seeds: normalized error 0.087 vs 0.032; onset probe 0.250 vs 0.621 on
train kits, 0.202 vs 0.318 on held-out kits.

Verdict. Every seed at weight 0.1 beats every seed at weight 0 on prediction
error, on the E2 and E1 state-side win rates, and on action content, with no
overlap between the two groups of three. The headline stands: a weak
action-reconstruction term restores action content to the state and improves
the world model's prediction at the same time. Kit-swap sensitivity is at
ceiling for both and does not separate.

Seed variance is worth knowing. The E2 win rate against the action-only model
swings from 0.64 to 0.92 across baseline seeds while the normalized error moves
only 0.077 to 0.105, so the win rate is the noisier statistic and normalized
error is the one to quote. Held-out content at 0.1 ranges 0.25 to 0.40; the
direction is safe, the magnitude is not.

This closes the option A question. Claims that remain single-seed: the 0.3 and
1.0 points of the sweep, the AO-JEPA comparison, and everything in E3 and E4.

### Follow-ups (2026-09-07, docs/followups.md, docs/followups/item1-5.md)
Five checks on the option A headline. E4 on the 0.1 model passes on every seed
(inverse-model onset F1 0.39-0.44 against controls at 0.25). A 40-epoch baseline
does not close the gap on any measure and sheds more action content than the
20-epoch one. The held-out predictor swap improves across three seeds (0.92-0.96
against 0.89), but E1 kit-swap sensitivity on unseen kits is identical for both
weights. A fixed-readout eval shows that under the training mask the predicted
grid scores at its ceiling for every model, so the scale-free prediction number
is the full-mask readout ratio: 0.92-1.00 for aux_rec 0.1 against 0.80-0.88 for
aux_rec 0. A mel-reconstruction control halves the normalized error but does not
move that ratio, and both regularized models enter a regime where error / teacher
variance rewards spread rather than prediction. Consequences: error / teacher
variance is retired for cross-model claims; the content gain is action-specific
on unseen kits only; the prediction gain is specific to the action target on the
readout ratio, pending mel-control seeds.

