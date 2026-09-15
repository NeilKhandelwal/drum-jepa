# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.2229 | 0.1438 | 0.118 [0.097, 0.143] | -0.4994 [-0.5419, -0.4587] |
| clean | ring_out | 42 | 0.1356 | 0.1510 | 0.310 [0.100, 0.438] | -0.2045 [-0.6412, +0.0768] |
| clean | dense | 4110 | 0.2302 | 0.1449 | 0.102 [0.085, 0.122] | -0.4967 [-0.5436, -0.4510] |
| clean | ring_out_le5 | 120 | 0.1998 | 0.1300 | 0.167 [0.050, 0.242] | -0.5452 [-0.7608, -0.4056] |
| kit_swap | all | 12204 | 0.2452 | 0.1438 | 0.083 [0.063, 0.109] | -0.6034 [-0.6462, -0.5594] |
| kit_swap | ring_out | 42 | 0.1464 | 0.1510 | 0.310 [0.100, 0.438] | -0.2770 [-0.7061, +0.0048] |
| kit_swap | dense | 4110 | 0.2543 | 0.1449 | 0.064 [0.052, 0.077] | -0.6068 [-0.6476, -0.5684] |
| kit_swap | ring_out_le5 | 120 | 0.2130 | 0.1300 | 0.167 [0.050, 0.242] | -0.6131 [-0.8333, -0.4724] |
| full_mask | all | 12204 | 2.4793 | 1.3998 | 0.027 [0.003, 0.070] | -0.5613 [-0.6098, -0.4978] |
| full_mask | ring_out | 42 | 2.0640 | 1.4218 | 0.167 [0.000, 0.375] | -0.3424 [-0.4343, -0.2474] |
| full_mask | dense | 4110 | 2.3964 | 1.3824 | 0.045 [0.001, 0.130] | -0.5300 [-0.6306, -0.4057] |
| full_mask | ring_out_le5 | 120 | 2.4655 | 1.4537 | 0.058 [0.000, 0.167] | -0.5106 [-0.5786, -0.4246] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.941, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.118, log-ratio -0.4994 [-0.5419, -0.4587] -> action-only better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.310, log-ratio -0.2045 [-0.6412, +0.0768] -> tie (CI spans 0). Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.027, log-ratio -0.5613 [-0.6098, -0.4978] -> action-only better; kit-swapped s_t, all: n=12204, win 0.083, log-ratio -0.6034 [-0.6462, -0.5594] -> action-only better. Q1 predicts the full model wins under a full mask.
