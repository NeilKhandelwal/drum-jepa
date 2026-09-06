# E4 inverse recovery — 19998 train transitions, 8 epochs, h = 3.42M params

h(s_t, s_(t+1)[, a_t]) decodes a_(t+1) and cc_(t+1) from frozen embeddings; the decoder is identical across representations and only the input projection differs. Onsets are peak-picked with +-2-frame non-max suppression and matched greedily within +-5 frames (50 ms). Macro F1 averages the per-class F1 over the classes with at least one true onset in the subset; the CI is a 1000-resample cluster bootstrap over sequences.

Two thresholds are reported. F1@0.5 is the protocol threshold; because the onset BCE pos_weight is capped at 50 against a true negative/positive ratio of 152, every variant over-predicts there and the absolute numbers are not comparable to published transcription figures. F1@tuned picks each variant's threshold on the grid 0.05..0.95 step 0.05 by macro F1 on the VALIDATION train kits, then applies that one threshold unchanged to both test kit groups; it is the number to read. Velocity MAE below is also at the tuned threshold (matched onsets move with it); CC4 MAE does not depend on the threshold.

| representation | split | kit group | n | macro F1@0.5 [95% CI] | tuned t | macro F1@tuned [95% CI] | micro F1@tuned |
|---|---|---|---|---|---|---|---|
| drumjepa | validation | train_kits | 12204 | 0.104 [0.088, 0.120] | 0.35 | 0.119 [0.103, 0.132] | 0.195 |
| drumjepa | validation | heldout_kits | 16272 | 0.109 [0.093, 0.124] | 0.35 | 0.110 [0.096, 0.122] | 0.170 |
| drumjepa | test | train_kits | 11760 | 0.103 [0.090, 0.110] | 0.35 | 0.118 [0.102, 0.126] | 0.212 |
| drumjepa | test | heldout_kits | 15680 | 0.112 [0.094, 0.119] | 0.35 | 0.119 [0.099, 0.127] | 0.188 |
| aojepa | validation | train_kits | 12204 | 0.174 [0.153, 0.192] | 0.55 | 0.177 [0.154, 0.194] | 0.308 |
| aojepa | validation | heldout_kits | 16272 | 0.162 [0.143, 0.177] | 0.55 | 0.170 [0.149, 0.186] | 0.303 |
| aojepa | test | train_kits | 11760 | 0.165 [0.151, 0.172] | 0.55 | 0.164 [0.150, 0.170] | 0.304 |
| aojepa | test | heldout_kits | 15680 | 0.162 [0.145, 0.169] | 0.55 | 0.165 [0.146, 0.173] | 0.295 |
| random | validation | train_kits | 12204 | 0.245 [0.227, 0.256] | 0.60 | 0.257 [0.236, 0.269] | 0.431 |
| random | validation | heldout_kits | 16272 | 0.207 [0.189, 0.218] | 0.60 | 0.218 [0.197, 0.231] | 0.378 |
| random | test | train_kits | 11760 | 0.245 [0.230, 0.252] | 0.60 | 0.251 [0.235, 0.259] | 0.438 |
| random | test | heldout_kits | 15680 | 0.209 [0.194, 0.217] | 0.60 | 0.215 [0.199, 0.223] | 0.386 |
| raw_mel | validation | train_kits | 12204 | 0.252 [0.235, 0.263] | 0.60 | 0.258 [0.237, 0.269] | 0.453 |
| raw_mel | validation | heldout_kits | 16272 | 0.215 [0.199, 0.226] | 0.60 | 0.224 [0.203, 0.236] | 0.408 |
| raw_mel | test | train_kits | 11760 | 0.250 [0.233, 0.259] | 0.60 | 0.250 [0.236, 0.256] | 0.457 |
| raw_mel | test | heldout_kits | 15680 | 0.222 [0.203, 0.230] | 0.60 | 0.221 [0.206, 0.229] | 0.411 |
| drumjepa_with_at | validation | train_kits | 12204 | 0.139 [0.115, 0.152] | 0.40 | 0.146 [0.124, 0.161] | 0.221 |
| drumjepa_with_at | validation | heldout_kits | 16272 | 0.145 [0.125, 0.158] | 0.40 | 0.145 [0.125, 0.158] | 0.209 |
| drumjepa_with_at | test | train_kits | 11760 | 0.122 [0.106, 0.127] | 0.40 | 0.126 [0.111, 0.133] | 0.232 |
| drumjepa_with_at | test | heldout_kits | 15680 | 0.128 [0.114, 0.132] | 0.40 | 0.129 [0.115, 0.136] | 0.225 |
| raw_mel_with_at | validation | train_kits | 12204 | 0.209 [0.190, 0.220] | 0.60 | 0.219 [0.197, 0.231] | 0.391 |
| raw_mel_with_at | validation | heldout_kits | 16272 | 0.195 [0.179, 0.206] | 0.60 | 0.215 [0.193, 0.228] | 0.379 |
| raw_mel_with_at | test | train_kits | 11760 | 0.199 [0.187, 0.205] | 0.60 | 0.201 [0.186, 0.208] | 0.374 |
| raw_mel_with_at | test | heldout_kits | 15680 | 0.190 [0.179, 0.196] | 0.60 | 0.197 [0.185, 0.203] | 0.366 |

Predicting no onsets at all scores 0.000 macro and micro F1 everywhere, so the constant baseline is only informative for velocity and CC4 (below).

Micro precision/recall at the tuned threshold, test split, train kits: drumjepa 0.122/0.787, aojepa 0.203/0.605, raw_mel 0.354/0.643, random 0.337/0.626.

## Per-class onset F1 at the tuned threshold — test split, train kits

| class | n true onsets | drumjepa (t=0.35) | aojepa (t=0.55) | raw_mel (t=0.60) | random (t=0.60) |
|---|---|---|---|---|---|
| kick | 42036 | 0.185 | 0.276 | 0.490 | 0.473 |
| snare | 45846 | 0.220 | 0.329 | 0.449 | 0.400 |
| rimshot | 9990 | 0.104 | 0.248 | 0.458 | 0.490 |
| crossstick | 4272 | 0.070 | 0.039 | 0.008 | 0.001 |
| tom_hi | 3984 | 0.100 | 0.137 | 0.163 | 0.194 |
| tom_mid | 2976 | 0.003 | 0.015 | 0.028 | 0.104 |
| tom_lo | 5454 | 0.119 | 0.167 | 0.300 | 0.303 |
| hh_closed | 50112 | 0.298 | 0.354 | 0.539 | 0.539 |
| hh_open | 5316 | 0.064 | 0.136 | 0.206 | 0.206 |
| hh_pedal | 17340 | 0.137 | 0.193 | 0.281 | 0.289 |
| crash1 | 1038 | 0.000 | 0.000 | 0.000 | 0.000 |
| crash2 | 204 | 0.000 | 0.000 | 0.000 | 0.000 |
| ride | 14148 | 0.309 | 0.381 | 0.562 | 0.520 |
| ride_bell | 1056 | 0.048 | 0.016 | 0.011 | 0.000 |

## Velocity and hi-hat pedal

Velocity MAE is over matched onsets only at the tuned threshold, in MIDI units (x127); its baseline predicts the train-split mean onset velocity at every true onset. CC4 MAE is over all frames; its baseline is the train-split mean CC4.

| representation | split | kit group | velocity MAE | (constant) | CC4 MAE | (constant) |
|---|---|---|---|---|---|---|
| drumjepa | validation | train_kits | 24.0 | 29.9 | 27.7 | 30.4 |
| drumjepa | validation | heldout_kits | 26.8 | 29.9 | 30.0 | 30.4 |
| drumjepa | test | train_kits | 23.6 | 29.5 | 20.7 | 26.5 |
| drumjepa | test | heldout_kits | 27.0 | 29.5 | 23.6 | 26.5 |
| aojepa | validation | train_kits | 20.8 | 29.9 | 27.0 | 30.4 |
| aojepa | validation | heldout_kits | 23.5 | 29.9 | 28.9 | 30.4 |
| aojepa | test | train_kits | 21.4 | 29.5 | 20.1 | 26.5 |
| aojepa | test | heldout_kits | 23.8 | 29.5 | 22.1 | 26.5 |
| random | validation | train_kits | 16.6 | 29.9 | 26.4 | 30.4 |
| random | validation | heldout_kits | 18.0 | 29.9 | 27.9 | 30.4 |
| random | test | train_kits | 17.1 | 29.5 | 19.8 | 26.5 |
| random | test | heldout_kits | 18.5 | 29.5 | 21.7 | 26.5 |
| raw_mel | validation | train_kits | 16.7 | 29.9 | 25.8 | 30.4 |
| raw_mel | validation | heldout_kits | 18.3 | 29.9 | 27.3 | 30.4 |
| raw_mel | test | train_kits | 17.5 | 29.5 | 19.5 | 26.5 |
| raw_mel | test | heldout_kits | 18.7 | 29.5 | 21.2 | 26.5 |
| drumjepa_with_at | validation | train_kits | 22.5 | 29.9 | 22.2 | 30.4 |
| drumjepa_with_at | validation | heldout_kits | 23.4 | 29.9 | 22.6 | 30.4 |
| drumjepa_with_at | test | train_kits | 22.8 | 29.5 | 17.7 | 26.5 |
| drumjepa_with_at | test | heldout_kits | 24.2 | 29.5 | 18.2 | 26.5 |
| raw_mel_with_at | validation | train_kits | 18.2 | 29.9 | 21.7 | 30.4 |
| raw_mel_with_at | validation | heldout_kits | 19.4 | 29.9 | 22.3 | 30.4 |
| raw_mel_with_at | test | train_kits | 19.3 | 29.5 | 17.0 | 26.5 |
| raw_mel_with_at | test | heldout_kits | 20.0 | 29.5 | 17.8 | 26.5 |

## Training

| variant | h params | best val loss | epoch | train time (s) |
|---|---|---|---|---|
| drumjepa | 3.42M | 0.7019 | 7 | 167 |
| aojepa | 3.42M | 0.6373 | 7 | 162 |
| random | 3.42M | 0.5626 | 7 | 162 |
| raw_mel | 3.45M | 0.5529 | 7 | 96 |
| drumjepa_with_at | 3.52M | 0.6479 | 7 | 172 |
| raw_mel_with_at | 3.55M | 0.5743 | 7 | 102 |

## Q1/E4

A state transition does carry the action, but how much of it survives depends on the encoder, not on the decoder: under an identical h, and with each variant's onset threshold tuned for macro F1 on the validation train kits, the test-split macro onset F1 on the six train kits is random 0.251 [0.235, 0.259] at t=0.60, raw_mel 0.250 [0.236, 0.256] at t=0.60, aojepa 0.164 [0.150, 0.170] at t=0.55, drumjepa 0.118 [0.102, 0.126] at t=0.35, against 0.000 for predicting no onsets, and every representation beats both constant baselines on velocity and pedal position (drum-JEPA: velocity MAE 23.6 against 29.5 MIDI units at matched onsets, CC4 MAE 20.7 against 26.5).
Drum-JEPA is last of the 4 and beats neither control (random 0.251, raw_mel 0.250), so by the bar in notes/decisions.md the number does not count in drum-JEPA's favour: on this task the JEPA objective makes the state encoder WORSE than the same architecture untrained. That is what an action-conditioned objective predicts, though: f is handed a_(t+1) at every step, so neither s_t nor s_(t+1) ever has to encode it, and 12 layers of training are free to throw the onset detail away.
The E5 comparison lands early and it lands hard: AO-JEPA, the same architecture on the same audio with the actions removed, scores 0.164 [0.150, 0.170] at t=0.55 against drum-JEPA's 0.118, so removing action conditioning buys the state encoder 0.045 of macro F1. E5 should be read against this row, not around it.
Held-out kits cost drum-JEPA -0.001 of macro F1 (0.119 against 0.118) against -0.002 (aojepa), +0.037 (random), +0.029 (raw_mel), so it is the flattest across the kit split — but flat at 0.118 is a floor effect, not evidence that it generalizes: the representations that carry action content are the ones with something to lose.
Handing h the previous action moves macro F1 by +0.008 (drumjepa), -0.048 (raw_mel), so groove continuation is not what carries these scores; the negative entries are an optimization artifact of the fixed 8-epoch budget (their validation loss is higher too, 0.574 against 0.553 (raw_mel)), not a finding.
