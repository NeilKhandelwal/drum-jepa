# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.2182 | 0.1438 | 0.094 [0.079, 0.113] | -0.4890 [-0.5097, -0.4653] |
| clean | ring_out | 42 | 0.2318 | 0.1510 | 0.214 [0.067, 0.310] | -0.5787 [-0.6841, -0.4761] |
| clean | dense | 4110 | 0.2195 | 0.1449 | 0.089 [0.068, 0.114] | -0.4735 [-0.5067, -0.4378] |
| clean | ring_out_le5 | 120 | 0.2025 | 0.1300 | 0.142 [0.056, 0.194] | -0.5460 [-0.6138, -0.4824] |
| kit_swap | all | 12204 | 0.4236 | 0.1438 | 0.020 [0.014, 0.029] | -1.0948 [-1.1346, -1.0586] |
| kit_swap | ring_out | 42 | 0.4571 | 0.1510 | 0.071 [0.000, 0.143] | -1.3487 [-1.5526, -1.1515] |
| kit_swap | dense | 4110 | 0.4266 | 0.1449 | 0.019 [0.012, 0.026] | -1.0727 [-1.1406, -1.0105] |
| kit_swap | ring_out_le5 | 120 | 0.4214 | 0.1300 | 0.033 [0.000, 0.061] | -1.2580 [-1.4329, -1.1532] |
| full_mask | all | 12204 | 0.9654 | 1.3998 | 0.987 [0.979, 0.993] | +0.3907 [+0.3514, +0.4375] |
| full_mask | ring_out | 42 | 0.8751 | 1.4218 | 0.952 [0.875, 1.000] | +0.5687 [+0.3060, +0.8233] |
| full_mask | dense | 4110 | 0.9422 | 1.3824 | 0.985 [0.972, 0.994] | +0.4027 [+0.3337, +0.4828] |
| full_mask | ring_out_le5 | 120 | 0.9893 | 1.4537 | 0.967 [0.931, 1.000] | +0.4292 [+0.2989, +0.5965] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.995, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.094, log-ratio -0.4890 [-0.5097, -0.4653] -> action-only better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.214, log-ratio -0.5787 [-0.6841, -0.4761] -> action-only better. Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.987, log-ratio +0.3907 [+0.3514, +0.4375] -> full better; kit-swapped s_t, all: n=12204, win 0.020, log-ratio -1.0948 [-1.1346, -1.0586] -> action-only better. Q1 predicts the full model wins under a full mask.
