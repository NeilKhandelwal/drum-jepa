# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.0622 | 0.1438 | 0.941 [0.923, 0.955] | +0.9712 [+0.9159, +1.0206] |
| clean | ring_out | 42 | 0.6304 | 0.1510 | 0.238 [0.028, 0.472] | -1.1664 [-1.6039, -0.7311] |
| clean | dense | 4110 | 0.0517 | 0.1449 | 0.960 [0.942, 0.975] | +1.0224 [+0.9611, +1.0808] |
| clean | ring_out_le5 | 120 | 0.2951 | 0.1300 | 0.567 [0.417, 0.659] | -0.1833 [-0.4883, +0.0032] |
| kit_swap | all | 12204 | 3.0125 | 0.1438 | 0.002 [0.000, 0.005] | -3.1614 [-3.2166, -3.0913] |
| kit_swap | ring_out | 42 | 2.3533 | 0.1510 | 0.000 [0.000, 0.000] | -3.1130 [-3.3699, -2.9662] |
| kit_swap | dense | 4110 | 3.0493 | 0.1449 | 0.000 [0.000, 0.000] | -3.1496 [-3.1941, -3.0971] |
| kit_swap | ring_out_le5 | 120 | 2.7577 | 0.1300 | 0.000 [0.000, 0.000] | -3.2671 [-3.3974, -3.1907] |
| full_mask | all | 12204 | 0.0867 | 1.3998 | 0.995 [0.991, 0.998] | +3.0242 [+2.9658, +3.0775] |
| full_mask | ring_out | 42 | 0.6494 | 1.4218 | 0.810 [0.690, 0.929] | +1.4150 [+0.9856, +1.8769] |
| full_mask | dense | 4110 | 0.0731 | 1.3824 | 1.000 [0.999, 1.000] | +3.0312 [+2.9509, +3.1051] |
| full_mask | ring_out_le5 | 120 | 0.3355 | 1.4537 | 0.917 [0.880, 0.980] | +2.2741 [+2.0595, +2.4682] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.998, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.941, log-ratio +0.9712 [+0.9159, +1.0206] -> full better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.238, log-ratio -1.1664 [-1.6039, -0.7311] -> action-only better. Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.995, log-ratio +3.0242 [+2.9658, +3.0775] -> full better; kit-swapped s_t, all: n=12204, win 0.002, log-ratio -3.1614 [-3.2166, -3.0913] -> action-only better. Q1 predicts the full model wins under a full mask.
