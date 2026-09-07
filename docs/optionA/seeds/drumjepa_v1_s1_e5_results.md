# E5 content probes — drumjepa_v1_s1 (epoch 19) vs no AO-JEPA

30000 train clips (240000 steps) fit on the 6 train kits; validation and test scored on 28476 and 27440 clips, split by kit group. Per-step feature = the 16 frequency tokens of a 250 ms step, mean-pooled (256-d); raw_mel = the mean of the step's 25 frames (229-d). Threshold 0.5 unless stated; the best-threshold column picks one threshold on validation/train kits by macro-F1 and applies it to test.

## Onset: which classes were struck in the step

| representation | probe | split | kit group | macro-F1 | macro-F1 @ best thr |
|---|---|---|---|---|---|
| drumjepa | linear | validation | train_kits | 0.254 | 0.394 |
| drumjepa | linear | validation | heldout_kits | 0.196 | 0.353 |
| drumjepa | linear | test | train_kits | 0.265 [0.249, 0.276] | 0.391 |
| drumjepa | linear | test | heldout_kits | 0.208 [0.189, 0.219] | 0.350 |
| drumjepa | mlp | validation | train_kits | 0.295 | 0.406 |
| drumjepa | mlp | validation | heldout_kits | 0.230 | 0.341 |
| drumjepa | mlp | test | train_kits | 0.312 [0.293, 0.322] | 0.399 |
| drumjepa | mlp | test | heldout_kits | 0.244 [0.223, 0.253] | 0.337 |
| random | linear | validation | train_kits | 0.330 | 0.426 |
| random | linear | validation | heldout_kits | 0.292 | 0.371 |
| random | linear | test | train_kits | 0.365 [0.333, 0.382] | 0.447 |
| random | linear | test | heldout_kits | 0.320 [0.288, 0.333] | 0.388 |
| random | mlp | validation | train_kits | 0.380 | 0.442 |
| random | mlp | validation | heldout_kits | 0.314 | 0.373 |
| random | mlp | test | train_kits | 0.413 [0.391, 0.424] | 0.456 |
| random | mlp | test | heldout_kits | 0.336 [0.314, 0.344] | 0.388 |
| raw_mel | linear | validation | train_kits | 0.553 | 0.607 |
| raw_mel | linear | validation | heldout_kits | 0.276 | 0.308 |
| raw_mel | linear | test | train_kits | 0.583 [0.555, 0.605] | 0.625 |
| raw_mel | linear | test | heldout_kits | 0.277 [0.248, 0.291] | 0.303 |
| raw_mel | mlp | validation | train_kits | 0.682 | 0.700 |
| raw_mel | mlp | validation | heldout_kits | 0.298 | 0.317 |
| raw_mel | mlp | test | train_kits | 0.694 [0.667, 0.717] | 0.708 |
| raw_mel | mlp | test | heldout_kits | 0.301 [0.273, 0.312] | 0.316 |
| prior (trivial) | - | validation | train_kits | 0.136 | - |
| prior (trivial) | - | validation | heldout_kits | 0.136 | - |
| prior (trivial) | - | test | train_kits | 0.130 | - |
| prior (trivial) | - | test | heldout_kits | 0.130 | - |

CI: 1000-resample cluster bootstrap over sequences, on test only. F1 is not a mean of per-clip values, so the per-sequence tp/fp/fn are summed under multinomial weights and the macro-F1 recomputed exactly, rather than reusing eval_e1.cluster_bootstrap. The trivial baseline predicts each class independently at its training positive rate, whose expected F1 is 2pq/(p+q) in closed form.

### Per-class F1, linear probe, test split, train kits

| class | train positive rate | drumjepa | random | raw_mel | prior |
|---|---|---|---|---|---|
| kick | 0.4047 | 0.515 | 0.733 | 0.775 | 0.399 |
| snare | 0.3792 | 0.531 | 0.560 | 0.761 | 0.358 |
| rimshot | 0.1073 | 0.600 | 0.705 | 0.837 | 0.104 |
| crossstick | 0.0623 | 0.007 | 0.081 | 0.434 | 0.051 |
| tom_hi | 0.0503 | 0.102 | 0.399 | 0.700 | 0.039 |
| tom_mid | 0.0231 | 0.025 | 0.232 | 0.697 | 0.024 |
| tom_lo | 0.0481 | 0.066 | 0.370 | 0.736 | 0.048 |
| hh_closed | 0.2669 | 0.703 | 0.728 | 0.783 | 0.334 |
| hh_open | 0.0592 | 0.060 | 0.154 | 0.436 | 0.057 |
| hh_pedal | 0.2661 | 0.223 | 0.332 | 0.417 | 0.217 |
| crash1 | 0.0182 | 0.053 | 0.039 | 0.152 | 0.014 |
| crash2 | 0.0078 | 0.028 | 0.000 | 0.173 | 0.003 |
| ride | 0.2289 | 0.725 | 0.635 | 0.782 | 0.159 |
| ride_bell | 0.0369 | 0.076 | 0.145 | 0.486 | 0.017 |

### Pooling check: 16 frequency tokens concatenated (4096-d), linear onset probe

On a seeded 2000-clip subsample of each split, so these numbers are comparable to each other but not to the table above.

| representation | features | validation train kits | test train kits |
|---|---|---|---|
| drumjepa | pooled | 0.265 | 0.269 |
| drumjepa | tokens | 0.506 | 0.531 |

## Velocity, timing and hi-hat position

Velocity and timing are scored only on (clip, step, class) triples that have an onset, with one regression per class; the MAE is pooled over classes. Timing is the frame of the first onset inside the 25-frame step. The baseline predicts the per-class training mean (the training mean for hi-hat).

| target | representation | probe | split | kit group | MAE | baseline MAE | n rows |
|---|---|---|---|---|---|---|---|
| velocity (MIDI vel) | drumjepa | linear | validation | train_kits | 19.473 | 24.605 | 189984 |
| velocity (MIDI vel) | drumjepa | linear | validation | heldout_kits | 21.335 | 24.605 | 253312 |
| velocity (MIDI vel) | drumjepa | linear | test | train_kits | 18.810 | 24.560 | 170676 |
| velocity (MIDI vel) | drumjepa | linear | test | heldout_kits | 20.063 | 24.560 | 227568 |
| velocity (MIDI vel) | drumjepa | mlp | validation | train_kits | 20.194 | 24.605 | 189984 |
| velocity (MIDI vel) | drumjepa | mlp | validation | heldout_kits | 23.731 | 24.605 | 253312 |
| velocity (MIDI vel) | drumjepa | mlp | test | train_kits | 19.522 | 24.560 | 170676 |
| velocity (MIDI vel) | drumjepa | mlp | test | heldout_kits | 22.579 | 24.560 | 227568 |
| velocity (MIDI vel) | random | linear | validation | train_kits | 17.872 | 24.605 | 189984 |
| velocity (MIDI vel) | random | linear | validation | heldout_kits | 20.137 | 24.605 | 253312 |
| velocity (MIDI vel) | random | linear | test | train_kits | 17.335 | 24.560 | 170676 |
| velocity (MIDI vel) | random | linear | test | heldout_kits | 19.315 | 24.560 | 227568 |
| velocity (MIDI vel) | random | mlp | validation | train_kits | 16.470 | 24.605 | 189984 |
| velocity (MIDI vel) | random | mlp | validation | heldout_kits | 19.598 | 24.605 | 253312 |
| velocity (MIDI vel) | random | mlp | test | train_kits | 15.683 | 24.560 | 170676 |
| velocity (MIDI vel) | random | mlp | test | heldout_kits | 18.419 | 24.560 | 227568 |
| velocity (MIDI vel) | raw_mel | linear | validation | train_kits | 16.468 | 24.605 | 189984 |
| velocity (MIDI vel) | raw_mel | linear | validation | heldout_kits | 24.038 | 24.605 | 253312 |
| velocity (MIDI vel) | raw_mel | linear | test | train_kits | 16.073 | 24.560 | 170676 |
| velocity (MIDI vel) | raw_mel | linear | test | heldout_kits | 24.447 | 24.560 | 227568 |
| velocity (MIDI vel) | raw_mel | mlp | validation | train_kits | 12.589 | 24.605 | 189984 |
| velocity (MIDI vel) | raw_mel | mlp | validation | heldout_kits | 22.992 | 24.605 | 253312 |
| velocity (MIDI vel) | raw_mel | mlp | test | train_kits | 12.183 | 24.560 | 170676 |
| velocity (MIDI vel) | raw_mel | mlp | test | heldout_kits | 22.967 | 24.560 | 227568 |
| timing (ms) | drumjepa | linear | validation | train_kits | 51.816 | 63.262 | 189984 |
| timing (ms) | drumjepa | linear | validation | heldout_kits | 52.918 | 63.262 | 253312 |
| timing (ms) | drumjepa | linear | test | train_kits | 50.146 | 60.307 | 170676 |
| timing (ms) | drumjepa | linear | test | heldout_kits | 50.725 | 60.307 | 227568 |
| timing (ms) | drumjepa | mlp | validation | train_kits | 50.224 | 63.262 | 189984 |
| timing (ms) | drumjepa | mlp | validation | heldout_kits | 52.915 | 63.262 | 253312 |
| timing (ms) | drumjepa | mlp | test | train_kits | 48.750 | 60.307 | 170676 |
| timing (ms) | drumjepa | mlp | test | heldout_kits | 51.303 | 60.307 | 227568 |
| timing (ms) | random | linear | validation | train_kits | 47.392 | 63.262 | 189984 |
| timing (ms) | random | linear | validation | heldout_kits | 49.795 | 63.262 | 253312 |
| timing (ms) | random | linear | test | train_kits | 45.156 | 60.307 | 170676 |
| timing (ms) | random | linear | test | heldout_kits | 47.014 | 60.307 | 227568 |
| timing (ms) | random | mlp | validation | train_kits | 44.901 | 63.262 | 189984 |
| timing (ms) | random | mlp | validation | heldout_kits | 48.772 | 63.262 | 253312 |
| timing (ms) | random | mlp | test | train_kits | 43.557 | 60.307 | 170676 |
| timing (ms) | random | mlp | test | heldout_kits | 46.358 | 60.307 | 227568 |
| timing (ms) | raw_mel | linear | validation | train_kits | 53.015 | 63.262 | 189984 |
| timing (ms) | raw_mel | linear | validation | heldout_kits | 63.257 | 63.262 | 253312 |
| timing (ms) | raw_mel | linear | test | train_kits | 51.553 | 60.307 | 170676 |
| timing (ms) | raw_mel | linear | test | heldout_kits | 60.752 | 60.307 | 227568 |
| timing (ms) | raw_mel | mlp | validation | train_kits | 45.837 | 63.262 | 189984 |
| timing (ms) | raw_mel | mlp | validation | heldout_kits | 66.774 | 63.262 | 253312 |
| timing (ms) | raw_mel | mlp | test | train_kits | 45.422 | 60.307 | 170676 |
| timing (ms) | raw_mel | mlp | test | heldout_kits | 65.036 | 60.307 | 227568 |
| hihat (CC4/127) | drumjepa | linear | validation | train_kits | 0.173 | 0.193 | 97632 |
| hihat (CC4/127) | drumjepa | linear | validation | heldout_kits | 0.198 | 0.193 | 130176 |
| hihat (CC4/127) | drumjepa | linear | test | train_kits | 0.145 | 0.179 | 94080 |
| hihat (CC4/127) | drumjepa | linear | test | heldout_kits | 0.185 | 0.179 | 125440 |
| hihat (CC4/127) | drumjepa | mlp | validation | train_kits | 0.175 | 0.193 | 97632 |
| hihat (CC4/127) | drumjepa | mlp | validation | heldout_kits | 0.197 | 0.193 | 130176 |
| hihat (CC4/127) | drumjepa | mlp | test | train_kits | 0.145 | 0.179 | 94080 |
| hihat (CC4/127) | drumjepa | mlp | test | heldout_kits | 0.176 | 0.179 | 125440 |
| hihat (CC4/127) | random | linear | validation | train_kits | 0.171 | 0.193 | 97632 |
| hihat (CC4/127) | random | linear | validation | heldout_kits | 0.173 | 0.193 | 130176 |
| hihat (CC4/127) | random | linear | test | train_kits | 0.143 | 0.179 | 94080 |
| hihat (CC4/127) | random | linear | test | heldout_kits | 0.145 | 0.179 | 125440 |
| hihat (CC4/127) | random | mlp | validation | train_kits | 0.168 | 0.193 | 97632 |
| hihat (CC4/127) | random | mlp | validation | heldout_kits | 0.173 | 0.193 | 130176 |
| hihat (CC4/127) | random | mlp | test | train_kits | 0.141 | 0.179 | 94080 |
| hihat (CC4/127) | random | mlp | test | heldout_kits | 0.147 | 0.179 | 125440 |
| hihat (CC4/127) | raw_mel | linear | validation | train_kits | 0.159 | 0.193 | 97632 |
| hihat (CC4/127) | raw_mel | linear | validation | heldout_kits | 0.184 | 0.193 | 130176 |
| hihat (CC4/127) | raw_mel | linear | test | train_kits | 0.132 | 0.179 | 94080 |
| hihat (CC4/127) | raw_mel | linear | test | heldout_kits | 0.168 | 0.179 | 125440 |
| hihat (CC4/127) | raw_mel | mlp | validation | train_kits | 0.146 | 0.193 | 97632 |
| hihat (CC4/127) | raw_mel | mlp | validation | heldout_kits | 0.183 | 0.193 | 130176 |
| hihat (CC4/127) | raw_mel | mlp | test | train_kits | 0.115 | 0.179 | 94080 |
| hihat (CC4/127) | raw_mel | mlp | test | heldout_kits | 0.164 | 0.179 | 125440 |

Velocity MAE is also 0.1481 (drumjepa), 0.1365 (random), 0.1266 (raw_mel) in velocity/127 units.

Classes with fewer than 200 training rows get no regression and contribute no rows: kick 97122, snare 91002, rimshot 25746, crossstick 14952, tom_hi 12084, tom_mid 5544, tom_lo 11550, hh_closed 64056, hh_open 14196, hh_pedal 63876, crash1 4368, crash2 1884, ride 54948, ride_bell 8868.

## Q1/E5

AO-JEPA was skipped in this run, so the drum-JEPA vs AO-JEPA comparison E5 exists for is not answered here; drum-JEPA's linear onset macro-F1 is 0.265 [0.249, 0.276].
Against the controls the trained encoder beats neither: random 0.365, raw_mel 0.583, trivial class-prior baseline 0.130; the best representation overall is raw_mel at 0.583, so the same warning as E3 applies — a probe only counts when it beats the spectrum it was computed from.
What is recoverable is which classes were struck, and only for the frequent, spectrally distinct ones: ride, hh_closed, rimshot score highest and crossstick, tom_mid, crash2 lowest under the drum-JEPA linear probe, tracking how often the class occurs rather than anything about the model; and mean-pooling the 16 frequency tokens of a step throws content away — concatenating them instead lifts drum-JEPA from 0.269 to 0.531 on the same clips, so every pooled number above is a lower bound.
What is not recoverable is the fine structure of the hit: velocity MAE is 18.8 MIDI units against 24.6 for predicting the per-class training mean, first-onset timing inside the 250 ms step is 50.1 ms against 60.3 ms, and the hi-hat pedal position is 0.145 against 0.179 (CC4/127) — every one of those beats its baseline, but by 10-25%, so the probes read the event far more sharply than its dynamics or its position inside the step.
Held-out kits cost the encoders little — drumjepa 0.208 against 0.265, random 0.320 against 0.365 — but cost the raw log-mel most of its lead (0.277 against 0.583), so the spectrum's advantage is largely kit-specific while what the encoders carry about the action is not.
