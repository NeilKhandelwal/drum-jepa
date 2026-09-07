# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.0767 | 0.1438 | 0.915 [0.902, 0.928] | +0.5515 [+0.5238, +0.5811] |
| clean | ring_out | 42 | 0.0886 | 0.1510 | 0.833 [0.750, 0.889] | +0.4017 [+0.1731, +0.6139] |
| clean | dense | 4110 | 0.0755 | 0.1449 | 0.940 [0.923, 0.955] | +0.5880 [+0.5451, +0.6367] |
| clean | ring_out_le5 | 120 | 0.0779 | 0.1300 | 0.833 [0.750, 0.869] | +0.4219 [+0.2594, +0.5336] |
| kit_swap | all | 12204 | 0.1275 | 0.1438 | 0.536 [0.499, 0.567] | +0.0487 [+0.0071, +0.0880] |
| kit_swap | ring_out | 42 | 0.1478 | 0.1510 | 0.381 [0.267, 0.467] | -0.2082 [-0.3695, -0.0992] |
| kit_swap | dense | 4110 | 0.1285 | 0.1449 | 0.554 [0.495, 0.608] | +0.0640 [+0.0059, +0.1203] |
| kit_swap | ring_out_le5 | 120 | 0.1323 | 0.1300 | 0.367 [0.292, 0.424] | -0.1413 [-0.2835, -0.0601] |
| full_mask | all | 12204 | 0.4790 | 1.3998 | 1.000 [1.000, 1.000] | +1.1013 [+1.0515, +1.1576] |
| full_mask | ring_out | 42 | 0.3757 | 1.4218 | 1.000 [1.000, 1.000] | +1.3902 [+1.1783, +1.5876] |
| full_mask | dense | 4110 | 0.4568 | 1.3824 | 1.000 [1.000, 1.000] | +1.1392 [+1.0428, +1.2407] |
| full_mask | ring_out_le5 | 120 | 0.4574 | 1.4537 | 1.000 [1.000, 1.000] | +1.2068 [+1.0818, +1.3451] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.996, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.915, log-ratio +0.5515 [+0.5238, +0.5811] -> full better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.833, log-ratio +0.4017 [+0.1731, +0.6139] -> full better. Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 1.000, log-ratio +1.1013 [+1.0515, +1.1576] -> full better; kit-swapped s_t, all: n=12204, win 0.536, log-ratio +0.0487 [+0.0071, +0.0880] -> full better. Q1 predicts the full model wins under a full mask.
