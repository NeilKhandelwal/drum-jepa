# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.0497 | 0.1438 | 0.978 [0.950, 0.993] | +1.3538 [+1.2573, +1.4276] |
| clean | ring_out | 42 | 0.5553 | 0.1510 | 0.167 [0.028, 0.312] | -1.1855 [-1.3864, -0.9590] |
| clean | dense | 4110 | 0.0309 | 0.1449 | 0.997 [0.993, 1.000] | +1.4934 [+1.4323, +1.5488] |
| clean | ring_out_le5 | 120 | 0.2524 | 0.1300 | 0.658 [0.530, 0.769] | +0.0392 [-0.2700, +0.3300] |
| kit_swap | all | 12204 | 3.0021 | 0.1438 | 0.001 [0.000, 0.003] | -3.1613 [-3.2199, -3.0790] |
| kit_swap | ring_out | 42 | 2.1439 | 0.1510 | 0.024 [0.000, 0.062] | -2.9688 [-3.3489, -2.6625] |
| kit_swap | dense | 4110 | 3.0585 | 0.1449 | 0.000 [0.000, 0.000] | -3.1566 [-3.1982, -3.1036] |
| kit_swap | ring_out_le5 | 120 | 2.6757 | 0.1300 | 0.008 [0.000, 0.019] | -3.2126 [-3.4194, -3.1100] |
| full_mask | all | 12204 | 0.0711 | 1.3998 | 0.993 [0.983, 0.998] | +3.3471 [+3.2349, +3.4301] |
| full_mask | ring_out | 42 | 0.5720 | 1.4218 | 0.857 [0.708, 1.000] | +1.3881 [+1.0550, +1.7212] |
| full_mask | dense | 4110 | 0.0502 | 1.3824 | 1.000 [0.999, 1.000] | +3.4067 [+3.2988, +3.4919] |
| full_mask | ring_out_le5 | 120 | 0.2830 | 1.4537 | 0.933 [0.889, 1.000] | +2.4779 [+2.2321, +2.7596] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.995, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.978, log-ratio +1.3538 [+1.2573, +1.4276] -> full better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.167, log-ratio -1.1855 [-1.3864, -0.9590] -> action-only better. Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.993, log-ratio +3.3471 [+3.2349, +3.4301] -> full better; kit-swapped s_t, all: n=12204, win 0.001, log-ratio -3.1613 [-3.2199, -3.0790] -> action-only better. Q1 predicts the full model wins under a full mask.
