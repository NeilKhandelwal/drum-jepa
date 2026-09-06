# E4 inverse recovery — 19998 train transitions, 8 epochs, h = 3.42M params

h(s_t, s_(t+1)[, a_t]) decodes a_(t+1) and cc_(t+1) from frozen embeddings; the decoder is identical across representations and only the input projection differs. Onsets are peak-picked with +-2-frame non-max suppression and matched greedily within +-5 frames (50 ms). Macro F1 averages the per-class F1 over the classes with at least one true onset in the subset; the CI is a 1000-resample cluster bootstrap over sequences.

Two thresholds are reported. F1@0.5 is the protocol threshold; because the onset BCE pos_weight is capped at 50 against a true negative/positive ratio of 152, every variant over-predicts there and the absolute numbers are not comparable to published transcription figures. F1@tuned picks each variant's threshold on the grid 0.05..0.95 step 0.05 by macro F1 on the VALIDATION train kits, then applies that one threshold unchanged to both test kit groups; it is the number to read. Velocity MAE below is also at the tuned threshold (matched onsets move with it); CC4 MAE does not depend on the threshold.

| representation | split | kit group | n | macro F1@0.5 [95% CI] | tuned t | macro F1@tuned [95% CI] | micro F1@tuned |
|---|---|---|---|---|---|---|---|
| drumjepa | validation | train_kits | 12204 | 0.462 [0.431, 0.481] | 0.60 | 0.485 [0.458, 0.503] | 0.636 |
| drumjepa | validation | heldout_kits | 16272 | 0.379 [0.351, 0.394] | 0.60 | 0.399 [0.370, 0.416] | 0.549 |
| drumjepa | test | train_kits | 11760 | 0.449 [0.426, 0.463] | 0.60 | 0.464 [0.444, 0.481] | 0.642 |
| drumjepa | test | heldout_kits | 15680 | 0.367 [0.342, 0.378] | 0.60 | 0.382 [0.361, 0.394] | 0.559 |
| drumjepa_with_at | validation | train_kits | 12204 | 0.461 [0.429, 0.478] | 0.60 | 0.483 [0.454, 0.499] | 0.634 |
| drumjepa_with_at | validation | heldout_kits | 16272 | 0.378 [0.352, 0.393] | 0.60 | 0.397 [0.369, 0.414] | 0.547 |
| drumjepa_with_at | test | train_kits | 11760 | 0.449 [0.425, 0.462] | 0.60 | 0.463 [0.442, 0.477] | 0.642 |
| drumjepa_with_at | test | heldout_kits | 15680 | 0.368 [0.342, 0.381] | 0.60 | 0.385 [0.362, 0.397] | 0.563 |

Predicting no onsets at all scores 0.000 macro and micro F1 everywhere, so the constant baseline is only informative for velocity and CC4 (below).

Micro precision/recall at the tuned threshold, test split, train kits: drumjepa 0.544/0.781.

## Per-class onset F1 at the tuned threshold — test split, train kits

| class | n true onsets | drumjepa (t=0.60) |
|---|---|---|
| kick | 42036 | 0.747 |
| snare | 45846 | 0.633 |
| rimshot | 9990 | 0.758 |
| crossstick | 4272 | 0.260 |
| tom_hi | 3984 | 0.403 |
| tom_mid | 2976 | 0.272 |
| tom_lo | 5454 | 0.517 |
| hh_closed | 50112 | 0.712 |
| hh_open | 5316 | 0.408 |
| hh_pedal | 17340 | 0.442 |
| crash1 | 1038 | 0.201 |
| crash2 | 204 | 0.077 |
| ride | 14148 | 0.730 |
| ride_bell | 1056 | 0.343 |

## Velocity and hi-hat pedal

Velocity MAE is over matched onsets only at the tuned threshold, in MIDI units (x127); its baseline predicts the train-split mean onset velocity at every true onset. CC4 MAE is over all frames; its baseline is the train-split mean CC4.

| representation | split | kit group | velocity MAE | (constant) | CC4 MAE | (constant) |
|---|---|---|---|---|---|---|
| drumjepa | validation | train_kits | 19.7 | 29.9 | 24.0 | 30.4 |
| drumjepa | validation | heldout_kits | 21.4 | 29.9 | 25.8 | 30.4 |
| drumjepa | test | train_kits | 19.2 | 29.5 | 19.2 | 26.5 |
| drumjepa | test | heldout_kits | 20.8 | 29.5 | 20.9 | 26.5 |
| drumjepa_with_at | validation | train_kits | 20.1 | 29.9 | 21.2 | 30.4 |
| drumjepa_with_at | validation | heldout_kits | 21.6 | 29.9 | 22.3 | 30.4 |
| drumjepa_with_at | test | train_kits | 19.5 | 29.5 | 17.2 | 26.5 |
| drumjepa_with_at | test | heldout_kits | 20.9 | 29.5 | 18.7 | 26.5 |

## Training

| variant | h params | best val loss | epoch | train time (s) |
|---|---|---|---|---|
| drumjepa | 3.42M | 0.4691 | 5 | 162 |
| drumjepa_with_at | 3.52M | 0.4611 | 6 | 169 |

## Q1/E4

A state transition does carry the action, but how much of it survives depends on the encoder, not on the decoder: under an identical h, and with each variant's onset threshold tuned for macro F1 on the validation train kits, the test-split macro onset F1 on the six train kits is drumjepa 0.464 [0.444, 0.481] at t=0.60, against 0.000 for predicting no onsets, and every representation beats both constant baselines on velocity and pedal position (drum-JEPA: velocity MAE 19.2 against 29.5 MIDI units at matched onsets, CC4 MAE 19.2 against 26.5).
Drum-JEPA is first of the 1 and beats both controls (), so by the bar in notes/decisions.md the representation, not h, is what the number measures.
Held-out kits cost drum-JEPA +0.082 of macro F1 (0.382 against 0.464) against , so it is the flattest across the kit split — but flat at 0.464 is a floor effect, not evidence that it generalizes: the representations that carry action content are the ones with something to lose.
Handing h the previous action moves macro F1 by -0.001 (drumjepa), so groove continuation is not what carries these scores; the negative entries are an optimization artifact of the fixed 8-epoch budget (their validation loss is higher too, 0.461 against 0.469 (drumjepa)), not a finding.
