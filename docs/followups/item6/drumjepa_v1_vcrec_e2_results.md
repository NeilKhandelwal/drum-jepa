# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.1756 | 0.1438 | 0.242 [0.226, 0.256] | -0.2413 [-0.2622, -0.2235] |
| clean | ring_out | 42 | 0.1758 | 0.1510 | 0.262 [0.194, 0.312] | -0.3989 [-0.5135, -0.2643] |
| clean | dense | 4110 | 0.1806 | 0.1449 | 0.237 [0.218, 0.256] | -0.2426 [-0.2669, -0.2193] |
| clean | ring_out_le5 | 120 | 0.1750 | 0.1300 | 0.167 [0.097, 0.203] | -0.4045 [-0.5203, -0.3484] |
| kit_swap | all | 12204 | 0.1958 | 0.1438 | 0.167 [0.151, 0.182] | -0.3591 [-0.3805, -0.3400] |
| kit_swap | ring_out | 42 | 0.1956 | 0.1510 | 0.262 [0.194, 0.312] | -0.5169 [-0.6596, -0.3982] |
| kit_swap | dense | 4110 | 0.2015 | 0.1449 | 0.155 [0.140, 0.175] | -0.3618 [-0.3876, -0.3335] |
| kit_swap | ring_out_le5 | 120 | 0.1892 | 0.1300 | 0.175 [0.098, 0.213] | -0.4938 [-0.6207, -0.4310] |
| full_mask | all | 12204 | 2.1467 | 1.3998 | 0.065 [0.023, 0.122] | -0.4042 [-0.4638, -0.3267] |
| full_mask | ring_out | 42 | 1.6650 | 1.4218 | 0.310 [0.133, 0.452] | +0.0015 [-0.2593, +0.2677] |
| full_mask | dense | 4110 | 2.0624 | 1.3824 | 0.089 [0.014, 0.215] | -0.3721 [-0.4811, -0.2346] |
| full_mask | ring_out_le5 | 120 | 2.1186 | 1.4537 | 0.117 [0.044, 0.202] | -0.3027 [-0.4256, -0.1302] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.953, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.242, log-ratio -0.2413 [-0.2622, -0.2235] -> action-only better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.262, log-ratio -0.3989 [-0.5135, -0.2643] -> action-only better. Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.065, log-ratio -0.4042 [-0.4638, -0.3267] -> action-only better; kit-swapped s_t, all: n=12204, win 0.167, log-ratio -0.3591 [-0.3805, -0.3400] -> action-only better. Q1 predicts the full model wins under a full mask.
