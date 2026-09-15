# E5 content probes — drumjepa_v2_12k (epoch 19) vs no AO-JEPA

30000 train clips (240000 steps) fit on the 6 train kits; validation and test scored on 40680 and 39200 clips, split by kit group. Per-step feature = the 16 frequency tokens of a 250 ms step, mean-pooled (256-d); raw_mel = the mean of the step's 25 frames (229-d). Threshold 0.5 unless stated; the best-threshold column picks one threshold on validation/train kits by macro-F1 and applies it to test.

## Onset: which classes were struck in the step

| representation | probe | split | kit group | macro-F1 | macro-F1 @ best thr |
|---|---|---|---|---|---|
| drumjepa | linear | validation | train_kits | 0.273 | 0.415 |
| drumjepa | linear | validation | heldout_kits | 0.204 | 0.236 |
| drumjepa | linear | test | train_kits | 0.295 [0.268, 0.307] | 0.399 |
| drumjepa | linear | test | heldout_kits | 0.195 [0.176, 0.205] | 0.223 |
| drumjepa | mlp | validation | train_kits | 0.271 | 0.397 |
| drumjepa | mlp | validation | heldout_kits | 0.121 | 0.191 |
| drumjepa | mlp | test | train_kits | 0.296 [0.265, 0.308] | 0.382 |
| drumjepa | mlp | test | heldout_kits | 0.127 [0.112, 0.133] | 0.192 |
| random | linear | validation | train_kits | 0.315 | 0.419 |
| random | linear | validation | heldout_kits | 0.301 | 0.382 |
| random | linear | test | train_kits | 0.343 [0.316, 0.357] | 0.430 |
| random | linear | test | heldout_kits | 0.332 [0.301, 0.346] | 0.398 |
| random | mlp | validation | train_kits | 0.360 | 0.426 |
| random | mlp | validation | heldout_kits | 0.332 | 0.388 |
| random | mlp | test | train_kits | 0.389 [0.366, 0.401] | 0.442 |
| random | mlp | test | heldout_kits | 0.358 [0.333, 0.369] | 0.408 |
| raw_mel | linear | validation | train_kits | 0.459 | 0.541 |
| raw_mel | linear | validation | heldout_kits | 0.280 | 0.337 |
| raw_mel | linear | test | train_kits | 0.491 [0.464, 0.514] | 0.556 |
| raw_mel | linear | test | heldout_kits | 0.284 [0.260, 0.295] | 0.333 |
| raw_mel | mlp | validation | train_kits | 0.609 | 0.642 |
| raw_mel | mlp | validation | heldout_kits | 0.322 | 0.355 |
| raw_mel | mlp | test | train_kits | 0.629 [0.602, 0.654] | 0.657 |
| raw_mel | mlp | test | heldout_kits | 0.328 [0.305, 0.338] | 0.360 |
| prior (trivial) | - | validation | train_kits | 0.136 | - |
| prior (trivial) | - | validation | heldout_kits | 0.136 | - |
| prior (trivial) | - | test | train_kits | 0.131 | - |
| prior (trivial) | - | test | heldout_kits | 0.131 | - |

CI: 1000-resample cluster bootstrap over sequences, on test only. F1 is not a mean of per-clip values, so the per-sequence tp/fp/fn are summed under multinomial weights and the macro-F1 recomputed exactly, rather than reusing eval_e1.cluster_bootstrap. The trivial baseline predicts each class independently at its training positive rate, whose expected F1 is 2pq/(p+q) in closed form.

### Per-class F1, linear probe, test split, train kits

| class | train positive rate | drumjepa | random | raw_mel | prior |
|---|---|---|---|---|---|
| kick | 0.4047 | 0.611 | 0.725 | 0.719 | 0.399 |
| snare | 0.3754 | 0.569 | 0.538 | 0.706 | 0.356 |
| rimshot | 0.1086 | 0.509 | 0.632 | 0.736 | 0.105 |
| crossstick | 0.0627 | 0.011 | 0.076 | 0.274 | 0.051 |
| tom_hi | 0.0473 | 0.267 | 0.371 | 0.602 | 0.038 |
| tom_mid | 0.0246 | 0.162 | 0.162 | 0.558 | 0.025 |
| tom_lo | 0.0506 | 0.271 | 0.374 | 0.629 | 0.050 |
| hh_closed | 0.2720 | 0.729 | 0.718 | 0.762 | 0.338 |
| hh_open | 0.0626 | 0.055 | 0.119 | 0.337 | 0.058 |
| hh_pedal | 0.2602 | 0.179 | 0.321 | 0.287 | 0.215 |
| crash1 | 0.0187 | 0.002 | 0.030 | 0.059 | 0.014 |
| crash2 | 0.0086 | 0.000 | 0.013 | 0.134 | 0.004 |
| ride | 0.2243 | 0.707 | 0.630 | 0.738 | 0.158 |
| ride_bell | 0.0363 | 0.060 | 0.091 | 0.340 | 0.017 |

### Pooling check: 16 frequency tokens concatenated (4096-d), linear onset probe

On a seeded 2000-clip subsample of each split, so these numbers are comparable to each other but not to the table above.

| representation | features | validation train kits | test train kits |
|---|---|---|---|
| drumjepa | pooled | 0.210 | 0.228 |
| drumjepa | tokens | 0.527 | 0.550 |

## Velocity, timing and hi-hat position

Velocity and timing are scored only on (clip, step, class) triples that have an onset, with one regression per class; the MAE is pooled over classes. Timing is the frame of the first onset inside the 25-frame step. The baseline predicts the per-class training mean (the training mean for hi-hat).

| target | representation | probe | split | kit group | MAE | baseline MAE | n rows |
|---|---|---|---|---|---|---|---|
| velocity (MIDI vel) | drumjepa | linear | validation | train_kits | 19.101 | 24.566 | 379968 |
| velocity (MIDI vel) | drumjepa | linear | validation | heldout_kits | 66.357 | 24.566 | 253312 |
| velocity (MIDI vel) | drumjepa | linear | test | train_kits | 18.905 | 24.513 | 341352 |
| velocity (MIDI vel) | drumjepa | linear | test | heldout_kits | 66.629 | 24.513 | 227568 |
| velocity (MIDI vel) | drumjepa | mlp | validation | train_kits | 19.583 | 24.566 | 379968 |
| velocity (MIDI vel) | drumjepa | mlp | validation | heldout_kits | 49.606 | 24.566 | 253312 |
| velocity (MIDI vel) | drumjepa | mlp | test | train_kits | 19.414 | 24.513 | 341352 |
| velocity (MIDI vel) | drumjepa | mlp | test | heldout_kits | 50.185 | 24.513 | 227568 |
| velocity (MIDI vel) | random | linear | validation | train_kits | 18.658 | 24.566 | 379968 |
| velocity (MIDI vel) | random | linear | validation | heldout_kits | 19.738 | 24.566 | 253312 |
| velocity (MIDI vel) | random | linear | test | train_kits | 17.866 | 24.513 | 341352 |
| velocity (MIDI vel) | random | linear | test | heldout_kits | 18.771 | 24.513 | 227568 |
| velocity (MIDI vel) | random | mlp | validation | train_kits | 18.323 | 24.566 | 379968 |
| velocity (MIDI vel) | random | mlp | validation | heldout_kits | 19.578 | 24.566 | 253312 |
| velocity (MIDI vel) | random | mlp | test | train_kits | 17.062 | 24.513 | 341352 |
| velocity (MIDI vel) | random | mlp | test | heldout_kits | 18.299 | 24.513 | 227568 |
| velocity (MIDI vel) | raw_mel | linear | validation | train_kits | 17.683 | 24.566 | 379968 |
| velocity (MIDI vel) | raw_mel | linear | validation | heldout_kits | 22.508 | 24.566 | 253312 |
| velocity (MIDI vel) | raw_mel | linear | test | train_kits | 17.376 | 24.513 | 341352 |
| velocity (MIDI vel) | raw_mel | linear | test | heldout_kits | 23.020 | 24.513 | 227568 |
| velocity (MIDI vel) | raw_mel | mlp | validation | train_kits | 13.472 | 24.566 | 379968 |
| velocity (MIDI vel) | raw_mel | mlp | validation | heldout_kits | 22.040 | 24.566 | 253312 |
| velocity (MIDI vel) | raw_mel | mlp | test | train_kits | 12.849 | 24.513 | 341352 |
| velocity (MIDI vel) | raw_mel | mlp | test | heldout_kits | 21.831 | 24.513 | 227568 |
| timing (ms) | drumjepa | linear | validation | train_kits | 54.058 | 63.256 | 379968 |
| timing (ms) | drumjepa | linear | validation | heldout_kits | 167.603 | 63.256 | 253312 |
| timing (ms) | drumjepa | linear | test | train_kits | 51.081 | 60.279 | 341352 |
| timing (ms) | drumjepa | linear | test | heldout_kits | 161.672 | 60.279 | 227568 |
| timing (ms) | drumjepa | mlp | validation | train_kits | 55.342 | 63.256 | 379968 |
| timing (ms) | drumjepa | mlp | validation | heldout_kits | 100.400 | 63.256 | 253312 |
| timing (ms) | drumjepa | mlp | test | train_kits | 52.015 | 60.279 | 341352 |
| timing (ms) | drumjepa | mlp | test | heldout_kits | 97.116 | 60.279 | 227568 |
| timing (ms) | random | linear | validation | train_kits | 48.126 | 63.256 | 379968 |
| timing (ms) | random | linear | validation | heldout_kits | 49.560 | 63.256 | 253312 |
| timing (ms) | random | linear | test | train_kits | 45.860 | 60.279 | 341352 |
| timing (ms) | random | linear | test | heldout_kits | 46.817 | 60.279 | 227568 |
| timing (ms) | random | mlp | validation | train_kits | 47.193 | 63.256 | 379968 |
| timing (ms) | random | mlp | validation | heldout_kits | 49.092 | 63.256 | 253312 |
| timing (ms) | random | mlp | test | train_kits | 45.768 | 60.279 | 341352 |
| timing (ms) | random | mlp | test | heldout_kits | 47.171 | 60.279 | 227568 |
| timing (ms) | raw_mel | linear | validation | train_kits | 54.462 | 63.256 | 379968 |
| timing (ms) | raw_mel | linear | validation | heldout_kits | 62.083 | 63.256 | 253312 |
| timing (ms) | raw_mel | linear | test | train_kits | 52.461 | 60.279 | 341352 |
| timing (ms) | raw_mel | linear | test | heldout_kits | 59.721 | 60.279 | 227568 |
| timing (ms) | raw_mel | mlp | validation | train_kits | 48.274 | 63.256 | 379968 |
| timing (ms) | raw_mel | mlp | validation | heldout_kits | 64.084 | 63.256 | 253312 |
| timing (ms) | raw_mel | mlp | test | train_kits | 47.476 | 60.279 | 341352 |
| timing (ms) | raw_mel | mlp | test | heldout_kits | 62.541 | 60.279 | 227568 |
| hihat (CC4/127) | drumjepa | linear | validation | train_kits | 0.172 | 0.194 | 195264 |
| hihat (CC4/127) | drumjepa | linear | validation | heldout_kits | 0.319 | 0.194 | 130176 |
| hihat (CC4/127) | drumjepa | linear | test | train_kits | 0.146 | 0.181 | 188160 |
| hihat (CC4/127) | drumjepa | linear | test | heldout_kits | 0.314 | 0.181 | 125440 |
| hihat (CC4/127) | drumjepa | mlp | validation | train_kits | 0.179 | 0.194 | 195264 |
| hihat (CC4/127) | drumjepa | mlp | validation | heldout_kits | 0.250 | 0.194 | 130176 |
| hihat (CC4/127) | drumjepa | mlp | test | train_kits | 0.158 | 0.181 | 188160 |
| hihat (CC4/127) | drumjepa | mlp | test | heldout_kits | 0.273 | 0.181 | 125440 |
| hihat (CC4/127) | random | linear | validation | train_kits | 0.173 | 0.194 | 195264 |
| hihat (CC4/127) | random | linear | validation | heldout_kits | 0.172 | 0.194 | 130176 |
| hihat (CC4/127) | random | linear | test | train_kits | 0.147 | 0.181 | 188160 |
| hihat (CC4/127) | random | linear | test | heldout_kits | 0.146 | 0.181 | 125440 |
| hihat (CC4/127) | random | mlp | validation | train_kits | 0.183 | 0.194 | 195264 |
| hihat (CC4/127) | random | mlp | validation | heldout_kits | 0.184 | 0.194 | 130176 |
| hihat (CC4/127) | random | mlp | test | train_kits | 0.161 | 0.181 | 188160 |
| hihat (CC4/127) | random | mlp | test | heldout_kits | 0.161 | 0.181 | 125440 |
| hihat (CC4/127) | raw_mel | linear | validation | train_kits | 0.165 | 0.194 | 195264 |
| hihat (CC4/127) | raw_mel | linear | validation | heldout_kits | 0.179 | 0.194 | 130176 |
| hihat (CC4/127) | raw_mel | linear | test | train_kits | 0.141 | 0.181 | 188160 |
| hihat (CC4/127) | raw_mel | linear | test | heldout_kits | 0.163 | 0.181 | 125440 |
| hihat (CC4/127) | raw_mel | mlp | validation | train_kits | 0.152 | 0.194 | 195264 |
| hihat (CC4/127) | raw_mel | mlp | validation | heldout_kits | 0.172 | 0.194 | 130176 |
| hihat (CC4/127) | raw_mel | mlp | test | train_kits | 0.125 | 0.181 | 188160 |
| hihat (CC4/127) | raw_mel | mlp | test | heldout_kits | 0.154 | 0.181 | 125440 |

Velocity MAE is also 0.1489 (drumjepa), 0.1407 (random), 0.1368 (raw_mel) in velocity/127 units.

Classes with fewer than 200 training rows get no regression and contribute no rows: kick 97128, snare 90096, rimshot 26064, crossstick 15048, tom_hi 11340, tom_mid 5904, tom_lo 12144, hh_closed 65268, hh_open 15024, hh_pedal 62448, crash1 4488, crash2 2052, ride 53820, ride_bell 8712.

## Q1/E5

AO-JEPA was skipped in this run, so the drum-JEPA vs AO-JEPA comparison E5 exists for is not answered here; drum-JEPA's linear onset macro-F1 is 0.295 [0.268, 0.307].
Against the controls the trained encoder beats neither: random 0.343, raw_mel 0.491, trivial class-prior baseline 0.131; the best representation overall is raw_mel at 0.491, so the same warning as E3 applies — a probe only counts when it beats the spectrum it was computed from.
What is recoverable is which classes were struck, and only for the frequent, spectrally distinct ones: hh_closed, ride, kick score highest and crash2, crash1, crossstick lowest under the drum-JEPA linear probe, tracking how often the class occurs rather than anything about the model; and mean-pooling the 16 frequency tokens of a step throws content away — concatenating them instead lifts drum-JEPA from 0.228 to 0.550 on the same clips, so every pooled number above is a lower bound.
What is not recoverable is the fine structure of the hit: velocity MAE is 18.9 MIDI units against 24.5 for predicting the per-class training mean, first-onset timing inside the 250 ms step is 51.1 ms against 60.3 ms, and the hi-hat pedal position is 0.146 against 0.181 (CC4/127) — every one of those beats its baseline, but by 10-25%, so the probes read the event far more sharply than its dynamics or its position inside the step.
Held-out kits cost the encoders little — drumjepa 0.195 against 0.295, random 0.332 against 0.343 — but cost the raw log-mel most of its lead (0.284 against 0.491), so the spectrum's advantage is largely kit-specific while what the encoders carry about the action is not.
