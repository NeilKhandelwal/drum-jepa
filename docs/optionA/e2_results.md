# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.7494 | 0.1438 | 0.005 [0.001, 0.011] | -1.7216 [-1.7870, -1.6483] |
| clean | ring_out | 42 | 0.2722 | 0.1510 | 0.190 [0.067, 0.271] | -0.9287 [-1.2085, -0.7639] |
| clean | dense | 4110 | 0.8469 | 0.1449 | 0.000 [0.000, 0.001] | -1.8165 [-1.9001, -1.7347] |
| clean | ring_out_le5 | 120 | 0.4306 | 0.1300 | 0.083 [0.000, 0.125] | -1.3200 [-1.5249, -1.1329] |
| kit_swap | all | 12204 | 0.7662 | 0.1438 | 0.005 [0.001, 0.010] | -1.7452 [-1.8100, -1.6705] |
| kit_swap | ring_out | 42 | 0.2784 | 0.1510 | 0.190 [0.067, 0.271] | -0.9520 [-1.2357, -0.7867] |
| kit_swap | dense | 4110 | 0.8651 | 0.1449 | 0.000 [0.000, 0.001] | -1.8397 [-1.9228, -1.7587] |
| kit_swap | ring_out_le5 | 120 | 0.4381 | 0.1300 | 0.083 [0.000, 0.125] | -1.3381 [-1.5446, -1.1492] |
| full_mask | all | 12204 | 2.4723 | 1.3998 | 0.010 [0.002, 0.020] | -0.5561 [-0.5963, -0.5057] |
| full_mask | ring_out | 42 | 2.4291 | 1.4218 | 0.071 [0.000, 0.188] | -0.5112 [-0.6488, -0.4102] |
| full_mask | dense | 4110 | 2.4875 | 1.3824 | 0.018 [0.002, 0.041] | -0.5718 [-0.6392, -0.4974] |
| full_mask | ring_out_le5 | 120 | 2.5355 | 1.4537 | 0.025 [0.000, 0.079] | -0.5459 [-0.5950, -0.4927] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.936, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.005, log-ratio -1.7216 [-1.7870, -1.6483] -> action-only better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.190, log-ratio -0.9287 [-1.2085, -0.7639] -> action-only better. Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.010, log-ratio -0.5561 [-0.5963, -0.5057] -> action-only better; kit-swapped s_t, all: n=12204, win 0.005, log-ratio -1.7452 [-1.8100, -1.6705] -> action-only better. Q1 predicts the full model wins under a full mask.
