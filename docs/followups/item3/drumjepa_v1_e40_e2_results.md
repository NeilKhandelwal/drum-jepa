# E2 state ablation — full (epoch 39) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.0972 | 0.1438 | 0.776 [0.757, 0.797] | +0.3207 [+0.2922, +0.3534] |
| clean | ring_out | 42 | 0.1623 | 0.1510 | 0.476 [0.389, 0.548] | -0.0491 [-0.2258, +0.1164] |
| clean | dense | 4110 | 0.0966 | 0.1449 | 0.812 [0.780, 0.843] | +0.3464 [+0.3030, +0.3945] |
| clean | ring_out_le5 | 120 | 0.1082 | 0.1300 | 0.642 [0.513, 0.701] | +0.2002 [+0.0557, +0.2998] |
| kit_swap | all | 12204 | 0.2648 | 0.1438 | 0.332 [0.309, 0.357] | -0.4440 [-0.5083, -0.3818] |
| kit_swap | ring_out | 42 | 0.4686 | 0.1510 | 0.071 [0.000, 0.125] | -1.2649 [-1.4596, -1.0136] |
| kit_swap | dense | 4110 | 0.2586 | 0.1449 | 0.350 [0.306, 0.392] | -0.3969 [-0.5050, -0.3083] |
| kit_swap | ring_out_le5 | 120 | 0.3719 | 0.1300 | 0.192 [0.106, 0.239] | -0.9024 [-1.2537, -0.7337] |
| full_mask | all | 12204 | 0.6670 | 1.3998 | 0.999 [0.998, 1.000] | +0.7832 [+0.7272, +0.8434] |
| full_mask | ring_out | 42 | 0.6099 | 1.4218 | 1.000 [1.000, 1.000] | +0.9569 [+0.6058, +1.2700] |
| full_mask | dense | 4110 | 0.6414 | 1.3824 | 0.999 [0.998, 1.000] | +0.8088 [+0.7088, +0.9098] |
| full_mask | ring_out_le5 | 120 | 0.7199 | 1.4537 | 1.000 [1.000, 1.000] | +0.7825 [+0.6471, +0.9379] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.989, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.776, log-ratio +0.3207 [+0.2922, +0.3534] -> full better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.476, log-ratio -0.0491 [-0.2258, +0.1164] -> tie (CI spans 0). Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.999, log-ratio +0.7832 [+0.7272, +0.8434] -> full better; kit-swapped s_t, all: n=12204, win 0.332, log-ratio -0.4440 [-0.5083, -0.3818] -> action-only better. Q1 predicts the full model wins under a full mask.
