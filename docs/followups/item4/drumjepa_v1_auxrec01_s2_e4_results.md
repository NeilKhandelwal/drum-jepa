# E4 inverse recovery — 19998 train transitions, 8 epochs, h = 3.42M params

h(s_t, s_(t+1)[, a_t]) decodes a_(t+1) and cc_(t+1) from frozen embeddings; the decoder is identical across representations and only the input projection differs. Onsets are peak-picked with +-2-frame non-max suppression and matched greedily within +-5 frames (50 ms). Macro F1 averages the per-class F1 over the classes with at least one true onset in the subset; the CI is a 1000-resample cluster bootstrap over sequences.

Two thresholds are reported. F1@0.5 is the protocol threshold; because the onset BCE pos_weight is capped at 50 against a true negative/positive ratio of 152, every variant over-predicts there and the absolute numbers are not comparable to published transcription figures. F1@tuned picks each variant's threshold on the grid 0.05..0.95 step 0.05 by macro F1 on the VALIDATION train kits, then applies that one threshold unchanged to both test kit groups; it is the number to read. Velocity MAE below is also at the tuned threshold (matched onsets move with it); CC4 MAE does not depend on the threshold.

| representation | split | kit group | n | macro F1@0.5 [95% CI] | tuned t | macro F1@tuned [95% CI] | micro F1@tuned |
|---|---|---|---|---|---|---|---|
| drumjepa | validation | train_kits | 12204 | 0.388 [0.365, 0.401] | 0.60 | 0.418 [0.389, 0.432] | 0.584 |
| drumjepa | validation | heldout_kits | 16272 | 0.268 [0.247, 0.280] | 0.60 | 0.273 [0.251, 0.286] | 0.448 |
| drumjepa | test | train_kits | 11760 | 0.368 [0.351, 0.381] | 0.60 | 0.388 [0.370, 0.400] | 0.584 |
| drumjepa | test | heldout_kits | 15680 | 0.272 [0.252, 0.279] | 0.60 | 0.275 [0.254, 0.283] | 0.458 |
| drumjepa_with_at | validation | train_kits | 12204 | 0.379 [0.353, 0.391] | 0.60 | 0.406 [0.375, 0.420] | 0.576 |
| drumjepa_with_at | validation | heldout_kits | 16272 | 0.274 [0.253, 0.287] | 0.60 | 0.280 [0.257, 0.293] | 0.451 |
| drumjepa_with_at | test | train_kits | 11760 | 0.361 [0.341, 0.371] | 0.60 | 0.378 [0.357, 0.389] | 0.582 |
| drumjepa_with_at | test | heldout_kits | 15680 | 0.275 [0.254, 0.282] | 0.60 | 0.280 [0.255, 0.290] | 0.464 |

Predicting no onsets at all scores 0.000 macro and micro F1 everywhere, so the constant baseline is only informative for velocity and CC4 (below).

Micro precision/recall at the tuned threshold, test split, train kits: drumjepa 0.472/0.766.

## Per-class onset F1 at the tuned threshold — test split, train kits

| class | n true onsets | drumjepa (t=0.60) |
|---|---|---|
| kick | 42036 | 0.767 |
| snare | 45846 | 0.586 |
| rimshot | 9990 | 0.667 |
| crossstick | 4272 | 0.173 |
| tom_hi | 3984 | 0.320 |
| tom_mid | 2976 | 0.235 |
| tom_lo | 5454 | 0.377 |
| hh_closed | 50112 | 0.616 |
| hh_open | 5316 | 0.321 |
| hh_pedal | 17340 | 0.395 |
| crash1 | 1038 | 0.004 |
| crash2 | 204 | 0.000 |
| ride | 14148 | 0.679 |
| ride_bell | 1056 | 0.287 |

## Velocity and hi-hat pedal

Velocity MAE is over matched onsets only at the tuned threshold, in MIDI units (x127); its baseline predicts the train-split mean onset velocity at every true onset. CC4 MAE is over all frames; its baseline is the train-split mean CC4.

| representation | split | kit group | velocity MAE | (constant) | CC4 MAE | (constant) |
|---|---|---|---|---|---|---|
| drumjepa | validation | train_kits | 16.9 | 29.9 | 24.7 | 30.4 |
| drumjepa | validation | heldout_kits | 20.6 | 29.9 | 28.9 | 30.4 |
| drumjepa | test | train_kits | 17.0 | 29.5 | 18.9 | 26.5 |
| drumjepa | test | heldout_kits | 21.0 | 29.5 | 24.8 | 26.5 |
| drumjepa_with_at | validation | train_kits | 17.4 | 29.9 | 21.2 | 30.4 |
| drumjepa_with_at | validation | heldout_kits | 21.2 | 29.9 | 23.7 | 30.4 |
| drumjepa_with_at | test | train_kits | 17.4 | 29.5 | 16.9 | 26.5 |
| drumjepa_with_at | test | heldout_kits | 22.1 | 29.5 | 20.2 | 26.5 |

## Training

| variant | h params | best val loss | epoch | train time (s) |
|---|---|---|---|---|
| drumjepa | 3.42M | 0.4599 | 7 | 353 |
| drumjepa_with_at | 3.52M | 0.4510 | 7 | 393 |

## Q1/E4

A state transition does carry the action, but how much of it survives depends on the encoder, not on the decoder: under an identical h, and with each variant's onset threshold tuned for macro F1 on the validation train kits, the test-split macro onset F1 on the six train kits is drumjepa 0.388 [0.370, 0.400] at t=0.60, against 0.000 for predicting no onsets, and every representation beats both constant baselines on velocity and pedal position (drum-JEPA: velocity MAE 17.0 against 29.5 MIDI units at matched onsets, CC4 MAE 18.9 against 26.5).
Drum-JEPA is first of the 1 and beats both controls (), so by the bar in notes/decisions.md the representation, not h, is what the number measures.
Held-out kits cost drum-JEPA +0.113 of macro F1 (0.275 against 0.388) against , so it is the flattest across the kit split — but flat at 0.388 is a floor effect, not evidence that it generalizes: the representations that carry action content are the ones with something to lose.
Handing h the previous action moves macro F1 by -0.009 (drumjepa), so groove continuation is not what carries these scores; the negative entries are an optimization artifact of the fixed 8-epoch budget (their validation loss is higher too, 0.451 against 0.460 (drumjepa)), not a finding.
