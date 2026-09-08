# Decisions on details the paper leaves unspecified

One entry per decision. Say what was chosen, why, and what could go wrong.

## Mel normalization (2026-09-05)
`SegmentPairs` reads `mel_mean`/`mel_std` from `<cache_dir>/stats.json` by default
(train split, 1M random frames). Chosen so no caller can forget to normalize.

Watch for: the stats belong to the cache build, not the run. Rebuilding the cache,
changing the train kit subset, or pointing at a different cache directory changes
the normalization silently, and checkpoints trained under one set of stats are not
comparable under another. Every run config must record the `mel_mean`/`mel_std`
actually used, and eval code must load the checkpoint's values rather than the
current `stats.json`.

## State tokenization (2026-09-05)
Patch 25 frames x 15 mel bins on the 200x229 log-mel, with the mel axis zero-padded
(after normalization) from 229 to 240 bins. Grid 8 (time) x 16 (freq) = 128 tokens.
The paper's "25x15 -> 16x16 = 256 tokens" does not divide 200x229; 128 tokens is the
closest faithful reading with a 25-frame time patch, which also aligns state and
action tokens in time (one column per 250 ms).

Watch for: half the paper's token count halves the masking granularity. If E1
perturbation win rates are weak, try a 12/13-frame time patch (16 time columns).

## Action tokenization (2026-09-05)
One token per 25-frame step over all 14 drum classes (8 tokens), per CLAUDE.md.
CC4 hi-hat position enters through Conv1d(kernel 25, stride 25) -> d_model, added
to the action token of the same time step. Configurable via `action_patch_frames`.

Watch for: 8 tokens is very few for an 8-layer encoder and a 75% mask (2 visible).
If action loss is trivially low or the action encoder collapses, drop to 10 frames
(20 tokens).

## Masking (2026-09-05)
I-JEPA multi-block masks on the 8x16 state grid for s_{t+1}, 75% masked to start;
prediction loss on masked positions only. Action target a_{t+1}: random 75% of the
8 tokens. Positional embeddings: fixed 2-D sin-cos (state), 1-D sin-cos (action),
as in I-JEPA.

## Predictor inputs (2026-09-05)
f sees all s_t tokens (student), the visible s_{t+1} tokens (student), and learned
mask tokens with positional embedding at masked positions. a_{t+1} for cross-attention
comes from the student action encoder with gradient; action tokens are K/V only.
Targets are teacher-encoder outputs after LayerNorm, stop-grad. EMA tau 0.95 per step.

## Overfit wiring test (2026-09-05)
32 fixed pairs, lr 6e-4 after a 20-step warmup, MPS bf16: loss 2.03 -> 0.03 by step
200 (loss_s 1.38 -> 0.026, loss_a 1.29 -> 0.011); s_tea_std stayed 0.59-0.78 and
a_tea_std recovered from 0.05 to 0.45. Loss does not reach exactly 0 because the EMA
teacher keeps moving and masks are resampled every step. Passed.

## E2 action-only baseline (2026-09-05)
Same architecture, data, masks and recipe as drumjepa_v1; the only change is that
f receives no s_t tokens (`use_state: false`), so its input is the masked s_{t+1}
half plus cross-attention to a_{t+1}. The visible 25% of s_{t+1} is kept on purpose:
it is the same information the full model gets, so the comparison isolates s_t.

Consequence: the action-only model is invariant to kit swap by construction, so
its kit-swap win rate is 0.5 exactly. E2 therefore compares the two models' errors
under shared masks per condition (clean held-out, kit-swapped s_t, post-ring-out
windows where a_{t+1} has few or no onsets) and additionally under a full mask,
where kit identity can only come from s_t. Q1 predicts: near tie on clean, full
model wins on ring-out and under full mask.

Watch for: at 75% mask the visible s_{t+1} tokens leak kit identity to both
models, which can hide the value of s_t. The full-mask comparison is the control.

## E3 representation and baselines (2026-09-05)
Clip embedding = mean over the 128 state tokens of the teacher (EMA) encoder's
output; student reported alongside. Probes are trained on train-split embeddings
and scored on validation and test. Two baselines bound the probe numbers: mean
log-mel of the clip (kit identity may be trivially linear in the raw spectrogram)
and a random-init encoder with the same architecture (what the architecture gives
for free). A probe result only counts if the trained encoder beats both.

## E5 AO-JEPA baseline (2026-09-06)
`use_action: false`: same state encoder, same masks and recipe, predictor f sees s_t
and the masked s_{t+1} but has no cross-attention (no action encoder, no g loss).
This is an I-JEPA-style audio JEPA with one step of temporal context, and it isolates
action conditioning exactly. Student params 9.7M vs 18.8M (f loses its cross-attn
blocks: 3.2M vs 4.8M; Ea and g are absent).

Watch for: fewer parameters and a single loss term make this an easier optimization
problem, not a harder one. If AO-JEPA's state loss ends lower than drum-JEPA's that
is expected (it predicts without a conditioning signal it has to learn to use), and
says nothing about representation quality. Only the probes and the E3 geometry
decide E5.

## Option A: action reconstruction regularizer (2026-09-06)
`aux_rec: 1.0` adds BCE(rec_head(pool_freq(s_t)), a_t > 0) to the loss, pos_weight 50,
everything else identical to drumjepa_v1. Tests whether the state can hold action
content without losing its prediction advantage (E4/E5 found drum-JEPA's state
carries less action content than an untrained encoder).

Success: E5 linear onset probe rises from 0.22 to at least the untrained encoder's
0.37, and the E2 clean win rate over the action-only baseline stays near 0.78.
Watch for: the head reads frequency-pooled tokens, so it can only push content into
the pooled subspace; if the probe does not move, try the un-pooled head before
raising the coefficient.

## Cross-model prediction errors (2026-09-06)
MSE in a model's own teacher space scales with that teacher's embedding spread.
E2's full-vs-action-only comparison was fair only because both teachers had
std ~1.1. Option A's teacher has std 1.89. Report error / mean per-dim teacher
variance, or within-model win rates (E1), whenever comparing across models.

## Control auxiliary target (2026-09-07)
`aux_target: mel` (configs/drumjepa_v1_melrec.yaml) keeps option A's head and weight
and swaps only its target: the same frequency-pooled s_t step vectors now predict the
window's own normalized log-mel, averaged over each step's 25 frames, under MSE.
The 229 un-padded bins are used, not the 240 padded ones: the padding exists for the
patch grid, and the un-padded target needs no reshaping of the batch. This is
docs/followups.md item 2, the control that separates "action content in the state
helps" from "any weak auxiliary term helps".

Weight: the action term is BCE with pos_weight 50 and the mel term is an MSE, so the
same coefficient is not automatically the same nudge. Measured at initialization over
8 real training batches with the seed-0 init: action BCE 0.967 (std 0.006 across
batches), mel MSE 0.914 (std 0.022). They are within 6%, which is close to the
batch-to-batch spread, so the weight stays 0.1 rather than the exactly matched 0.106,
and the control differs from drumjepa_v1_auxrec01 in the target alone. For reference,
drumjepa_v1_auxrec01's recorded loss_rec is 0.921 at step 50 and 0.570 averaged over
epoch 0 (runs/drumjepa_v1_auxrec01/metrics.csv).

Watch for: matched at initialization is not matched throughout. The BCE term falls
fast once the head learns the onset prior, and a mel term that stays near 0.9 while
the action term reaches 0.5 is a stronger pull late in training, not an equal one.
Compare the two runs' loss_rec curves before reading the result; if they diverge by
more than a factor of two, the control is at a different effective weight and a
rerun at a rescaled weight decides it.
