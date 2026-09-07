# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.1169 | 0.1438 | 0.635 [0.605, 0.666] | +0.1430 [+0.1124, +0.1753] |
| clean | ring_out | 42 | 0.1303 | 0.1510 | 0.476 [0.310, 0.595] | +0.0386 [-0.1898, +0.2837] |
| clean | dense | 4110 | 0.1140 | 0.1449 | 0.700 [0.641, 0.748] | +0.1954 [+0.1413, +0.2444] |
| clean | ring_out_le5 | 120 | 0.1113 | 0.1300 | 0.525 [0.424, 0.608] | +0.0868 [-0.0345, +0.2261] |
| kit_swap | all | 12204 | 0.1471 | 0.1438 | 0.411 [0.386, 0.438] | -0.0916 [-0.1190, -0.0647] |
| kit_swap | ring_out | 42 | 0.1948 | 0.1510 | 0.238 [0.119, 0.357] | -0.4713 [-0.5715, -0.3463] |
| kit_swap | dense | 4110 | 0.1441 | 0.1449 | 0.455 [0.407, 0.497] | -0.0453 [-0.0930, -0.0052] |
| kit_swap | ring_out_le5 | 120 | 0.1557 | 0.1300 | 0.300 [0.208, 0.411] | -0.2767 [-0.3592, -0.1849] |
| full_mask | all | 12204 | 0.6121 | 1.3998 | 1.000 [1.000, 1.000] | +0.8468 [+0.8057, +0.8934] |
| full_mask | ring_out | 42 | 0.4951 | 1.4218 | 1.000 [1.000, 1.000] | +1.1114 [+0.8898, +1.3086] |
| full_mask | dense | 4110 | 0.5865 | 1.3824 | 1.000 [0.999, 1.000] | +0.8777 [+0.8046, +0.9531] |
| full_mask | ring_out_le5 | 120 | 0.6058 | 1.4537 | 1.000 [1.000, 1.000] | +0.9277 [+0.8084, +1.1168] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.994, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.635, log-ratio +0.1430 [+0.1124, +0.1753] -> full better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.476, log-ratio +0.0386 [-0.1898, +0.2837] -> tie (CI spans 0). Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 1.000, log-ratio +0.8468 [+0.8057, +0.8934] -> full better; kit-swapped s_t, all: n=12204, win 0.411, log-ratio -0.0916 [-0.1190, -0.0647] -> action-only better. Q1 predicts the full model wins under a full mask.
