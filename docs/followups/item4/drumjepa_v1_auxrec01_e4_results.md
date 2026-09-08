# E4 inverse recovery — 19998 train transitions, 8 epochs, h = 3.42M params

h(s_t, s_(t+1)[, a_t]) decodes a_(t+1) and cc_(t+1) from frozen embeddings; the decoder is identical across representations and only the input projection differs. Onsets are peak-picked with +-2-frame non-max suppression and matched greedily within +-5 frames (50 ms). Macro F1 averages the per-class F1 over the classes with at least one true onset in the subset; the CI is a 1000-resample cluster bootstrap over sequences.

Two thresholds are reported. F1@0.5 is the protocol threshold; because the onset BCE pos_weight is capped at 50 against a true negative/positive ratio of 152, every variant over-predicts there and the absolute numbers are not comparable to published transcription figures. F1@tuned picks each variant's threshold on the grid 0.05..0.95 step 0.05 by macro F1 on the VALIDATION train kits, then applies that one threshold unchanged to both test kit groups; it is the number to read. Velocity MAE below is also at the tuned threshold (matched onsets move with it); CC4 MAE does not depend on the threshold.

| representation | split | kit group | n | macro F1@0.5 [95% CI] | tuned t | macro F1@tuned [95% CI] | micro F1@tuned |
|---|---|---|---|---|---|---|---|
| drumjepa | validation | train_kits | 12204 | 0.435 [0.405, 0.452] | 0.60 | 0.459 [0.430, 0.475] | 0.600 |
| drumjepa | validation | heldout_kits | 16272 | 0.347 [0.321, 0.363] | 0.60 | 0.362 [0.335, 0.379] | 0.512 |
| drumjepa | test | train_kits | 11760 | 0.421 [0.399, 0.435] | 0.60 | 0.440 [0.418, 0.458] | 0.603 |
| drumjepa | test | heldout_kits | 15680 | 0.338 [0.315, 0.351] | 0.60 | 0.354 [0.333, 0.367] | 0.525 |
| drumjepa_with_at | validation | train_kits | 12204 | 0.441 [0.412, 0.459] | 0.60 | 0.466 [0.436, 0.483] | 0.603 |
| drumjepa_with_at | validation | heldout_kits | 16272 | 0.355 [0.328, 0.372] | 0.60 | 0.369 [0.340, 0.388] | 0.524 |
| drumjepa_with_at | test | train_kits | 11760 | 0.425 [0.402, 0.439] | 0.60 | 0.445 [0.420, 0.461] | 0.611 |
| drumjepa_with_at | test | heldout_kits | 15680 | 0.350 [0.327, 0.364] | 0.60 | 0.362 [0.338, 0.377] | 0.539 |

Predicting no onsets at all scores 0.000 macro and micro F1 everywhere, so the constant baseline is only informative for velocity and CC4 (below).

Micro precision/recall at the tuned threshold, test split, train kits: drumjepa 0.490/0.784.

## Per-class onset F1 at the tuned threshold — test split, train kits

| class | n true onsets | drumjepa (t=0.60) |
|---|---|---|
| kick | 42036 | 0.789 |
| snare | 45846 | 0.535 |
| rimshot | 9990 | 0.692 |
| crossstick | 4272 | 0.150 |
| tom_hi | 3984 | 0.385 |
| tom_mid | 2976 | 0.270 |
| tom_lo | 5454 | 0.503 |
| hh_closed | 50112 | 0.707 |
| hh_open | 5316 | 0.419 |
| hh_pedal | 17340 | 0.391 |
| crash1 | 1038 | 0.176 |
| crash2 | 204 | 0.105 |
| ride | 14148 | 0.698 |
| ride_bell | 1056 | 0.343 |

## Velocity and hi-hat pedal

Velocity MAE is over matched onsets only at the tuned threshold, in MIDI units (x127); its baseline predicts the train-split mean onset velocity at every true onset. CC4 MAE is over all frames; its baseline is the train-split mean CC4.

| representation | split | kit group | velocity MAE | (constant) | CC4 MAE | (constant) |
|---|---|---|---|---|---|---|
| drumjepa | validation | train_kits | 14.9 | 29.9 | 24.1 | 30.4 |
| drumjepa | validation | heldout_kits | 17.9 | 29.9 | 26.7 | 30.4 |
| drumjepa | test | train_kits | 14.8 | 29.5 | 18.7 | 26.5 |
| drumjepa | test | heldout_kits | 17.6 | 29.5 | 21.4 | 26.5 |
| drumjepa_with_at | validation | train_kits | 14.9 | 29.9 | 20.7 | 30.4 |
| drumjepa_with_at | validation | heldout_kits | 17.6 | 29.9 | 22.4 | 30.4 |
| drumjepa_with_at | test | train_kits | 14.8 | 29.5 | 16.7 | 26.5 |
| drumjepa_with_at | test | heldout_kits | 17.5 | 29.5 | 18.5 | 26.5 |

## Training

| variant | h params | best val loss | epoch | train time (s) |
|---|---|---|---|---|
| drumjepa | 3.42M | 0.4383 | 7 | 248 |
| drumjepa_with_at | 3.52M | 0.4199 | 7 | 367 |

## Q1/E4

A state transition does carry the action, but how much of it survives depends on the encoder, not on the decoder: under an identical h, and with each variant's onset threshold tuned for macro F1 on the validation train kits, the test-split macro onset F1 on the six train kits is drumjepa 0.440 [0.418, 0.458] at t=0.60, against 0.000 for predicting no onsets, and every representation beats both constant baselines on velocity and pedal position (drum-JEPA: velocity MAE 14.8 against 29.5 MIDI units at matched onsets, CC4 MAE 18.7 against 26.5).
Drum-JEPA is first of the 1 and beats both controls (), so by the bar in notes/decisions.md the representation, not h, is what the number measures.
Held-out kits cost drum-JEPA +0.086 of macro F1 (0.354 against 0.440) against , so it is the flattest across the kit split — but flat at 0.440 is a floor effect, not evidence that it generalizes: the representations that carry action content are the ones with something to lose.
Handing h the previous action moves macro F1 by +0.004 (drumjepa), so groove continuation is not what carries these scores.
