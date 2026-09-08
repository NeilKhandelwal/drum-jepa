# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.0293 | 0.1438 | 0.956 [0.934, 0.972] | +2.0521 [+1.8887, +2.1896] |
| clean | ring_out | 42 | 0.3113 | 0.1510 | 0.429 [0.286, 0.583] | -0.4047 [-0.9123, +0.0867] |
| clean | dense | 4110 | 0.0302 | 0.1449 | 0.954 [0.923, 0.982] | +2.0382 [+1.8208, +2.2369] |
| clean | ring_out_le5 | 120 | 0.1591 | 0.1300 | 0.733 [0.667, 0.813] | +0.5860 [+0.3215, +0.8281] |
| kit_swap | all | 12204 | 2.0572 | 0.1438 | 0.002 [0.000, 0.006] | -2.7872 [-2.8371, -2.7257] |
| kit_swap | ring_out | 42 | 1.5542 | 0.1510 | 0.024 [0.000, 0.062] | -2.6636 [-3.0278, -2.3974] |
| kit_swap | dense | 4110 | 2.0733 | 0.1449 | 0.000 [0.000, 0.001] | -2.7684 [-2.8102, -2.7136] |
| kit_swap | ring_out_le5 | 120 | 1.8641 | 0.1300 | 0.008 [0.000, 0.019] | -2.8691 [-3.0210, -2.8007] |
| full_mask | all | 12204 | 0.0351 | 1.3998 | 0.998 [0.997, 0.999] | +4.2881 [+4.1301, +4.4111] |
| full_mask | ring_out | 42 | 0.3369 | 1.4218 | 0.929 [0.861, 0.979] | +2.2741 [+1.6360, +2.9122] |
| full_mask | dense | 4110 | 0.0341 | 1.3824 | 1.000 [0.999, 1.000] | +4.2491 [+4.0118, +4.4453] |
| full_mask | ring_out_le5 | 120 | 0.1858 | 1.4537 | 0.958 [0.917, 0.992] | +3.1165 [+2.8694, +3.4249] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.998, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.956, log-ratio +2.0521 [+1.8887, +2.1896] -> full better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.429, log-ratio -0.4047 [-0.9123, +0.0867] -> tie (CI spans 0). Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.998, log-ratio +4.2881 [+4.1301, +4.4111] -> full better; kit-swapped s_t, all: n=12204, win 0.002, log-ratio -2.7872 [-2.8371, -2.7257] -> action-only better. Q1 predicts the full model wins under a full mask.
