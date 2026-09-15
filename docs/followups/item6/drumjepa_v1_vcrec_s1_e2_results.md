# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.1724 | 0.1438 | 0.279 [0.251, 0.306] | -0.2167 [-0.2455, -0.1853] |
| clean | ring_out | 42 | 0.1856 | 0.1510 | 0.286 [0.100, 0.389] | -0.3771 [-0.7817, -0.1394] |
| clean | dense | 4110 | 0.1761 | 0.1449 | 0.284 [0.246, 0.324] | -0.2109 [-0.2466, -0.1705] |
| clean | ring_out_le5 | 120 | 0.1615 | 0.1300 | 0.217 [0.128, 0.260] | -0.3146 [-0.4711, -0.2178] |
| kit_swap | all | 12204 | 0.1769 | 0.1438 | 0.251 [0.223, 0.278] | -0.2464 [-0.2752, -0.2153] |
| kit_swap | ring_out | 42 | 0.1924 | 0.1510 | 0.286 [0.100, 0.389] | -0.4203 [-0.7944, -0.1914] |
| kit_swap | dense | 4110 | 0.1810 | 0.1449 | 0.253 [0.215, 0.293] | -0.2435 [-0.2810, -0.2009] |
| kit_swap | ring_out_le5 | 120 | 0.1661 | 0.1300 | 0.217 [0.128, 0.260] | -0.3451 [-0.4970, -0.2462] |
| full_mask | all | 12204 | 2.5224 | 1.3998 | 0.053 [0.016, 0.103] | -0.5608 [-0.6274, -0.4766] |
| full_mask | ring_out | 42 | 1.9113 | 1.4218 | 0.214 [0.033, 0.389] | -0.1293 [-0.3904, +0.1562] |
| full_mask | dense | 4110 | 2.4488 | 1.3824 | 0.069 [0.008, 0.174] | -0.5377 [-0.6623, -0.3835] |
| full_mask | ring_out_le5 | 120 | 2.3662 | 1.4537 | 0.092 [0.024, 0.185] | -0.4089 [-0.5447, -0.2237] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.853, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.279, log-ratio -0.2167 [-0.2455, -0.1853] -> action-only better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.286, log-ratio -0.3771 [-0.7817, -0.1394] -> action-only better. Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.053, log-ratio -0.5608 [-0.6274, -0.4766] -> action-only better; kit-swapped s_t, all: n=12204, win 0.251, log-ratio -0.2464 [-0.2752, -0.2153] -> action-only better. Q1 predicts the full model wins under a full mask.
