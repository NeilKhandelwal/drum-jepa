# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.0673 | 0.1438 | 0.931 [0.910, 0.950] | +0.7200 [+0.6764, +0.7624] |
| clean | ring_out | 42 | 0.2833 | 0.1510 | 0.262 [0.125, 0.381] | -0.4159 [-0.8075, +0.0148] |
| clean | dense | 4110 | 0.0634 | 0.1449 | 0.953 [0.931, 0.969] | +0.7582 [+0.7011, +0.8181] |
| clean | ring_out_le5 | 120 | 0.1740 | 0.1300 | 0.542 [0.417, 0.667] | +0.0022 [-0.2463, +0.2183] |
| kit_swap | all | 12204 | 1.7335 | 0.1438 | 0.001 [0.000, 0.003] | -2.6141 [-2.6622, -2.5548] |
| kit_swap | ring_out | 42 | 1.4291 | 0.1510 | 0.024 [0.000, 0.062] | -2.5758 [-2.9342, -2.2737] |
| kit_swap | dense | 4110 | 1.7525 | 0.1449 | 0.000 [0.000, 0.001] | -2.5954 [-2.6355, -2.5443] |
| kit_swap | ring_out_le5 | 120 | 1.5999 | 0.1300 | 0.008 [0.000, 0.019] | -2.7067 [-2.8972, -2.5914] |
| full_mask | all | 12204 | 0.1642 | 1.3998 | 0.999 [0.998, 1.000] | +2.2150 [+2.1565, +2.2831] |
| full_mask | ring_out | 42 | 0.3577 | 1.4218 | 0.976 [0.917, 1.000] | +1.7150 [+1.3248, +2.0927] |
| full_mask | dense | 4110 | 0.1620 | 1.3824 | 1.000 [0.999, 1.000] | +2.2039 [+2.1147, +2.2883] |
| full_mask | ring_out_le5 | 120 | 0.2732 | 1.4537 | 0.983 [0.956, 1.000] | +1.9088 [+1.7048, +2.1047] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.998, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.931, log-ratio +0.7200 [+0.6764, +0.7624] -> full better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.262, log-ratio -0.4159 [-0.8075, +0.0148] -> tie (CI spans 0). Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.999, log-ratio +2.2150 [+2.1565, +2.2831] -> full better; kit-swapped s_t, all: n=12204, win 0.001, log-ratio -2.6141 [-2.6622, -2.5548] -> action-only better. Q1 predicts the full model wins under a full mask.
