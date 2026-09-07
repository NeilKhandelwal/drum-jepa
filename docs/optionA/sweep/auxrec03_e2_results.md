# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.4471 | 0.1438 | 0.013 [0.004, 0.028] | -1.2383 [-1.2887, -1.1782] |
| clean | ring_out | 42 | 0.2434 | 0.1510 | 0.190 [0.067, 0.271] | -0.8327 [-1.1109, -0.6810] |
| clean | dense | 4110 | 0.4804 | 0.1449 | 0.001 [0.000, 0.002] | -1.2779 [-1.3354, -1.2109] |
| clean | ring_out_le5 | 120 | 0.3215 | 0.1300 | 0.075 [0.000, 0.109] | -1.0964 [-1.2399, -0.9814] |
| kit_swap | all | 12204 | 0.5449 | 0.1438 | 0.008 [0.002, 0.020] | -1.4406 [-1.4971, -1.3762] |
| kit_swap | ring_out | 42 | 0.2931 | 0.1510 | 0.167 [0.033, 0.241] | -1.0139 [-1.3625, -0.8223] |
| kit_swap | dense | 4110 | 0.5927 | 0.1449 | 0.000 [0.000, 0.002] | -1.4955 [-1.5492, -1.4369] |
| kit_swap | ring_out_le5 | 120 | 0.3774 | 0.1300 | 0.067 [0.000, 0.101] | -1.2532 [-1.4582, -1.1056] |
| full_mask | all | 12204 | 1.3706 | 1.3998 | 0.526 [0.422, 0.638] | +0.0402 [-0.0144, +0.1024] |
| full_mask | ring_out | 42 | 1.3452 | 1.4218 | 0.452 [0.200, 0.625] | +0.0913 [-0.0301, +0.1692] |
| full_mask | dense | 4110 | 1.3502 | 1.3824 | 0.538 [0.387, 0.688] | +0.0467 [-0.0384, +0.1342] |
| full_mask | ring_out_le5 | 120 | 1.4685 | 1.4537 | 0.367 [0.205, 0.510] | +0.0107 [-0.0537, +0.0807] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.998, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.013, log-ratio -1.2383 [-1.2887, -1.1782] -> action-only better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.190, log-ratio -0.8327 [-1.1109, -0.6810] -> action-only better. Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.526, log-ratio +0.0402 [-0.0144, +0.1024] -> tie (CI spans 0); kit-swapped s_t, all: n=12204, win 0.008, log-ratio -1.4406 [-1.4971, -1.3762] -> action-only better. Q1 predicts the full model wins under a full mask.
