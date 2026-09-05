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
