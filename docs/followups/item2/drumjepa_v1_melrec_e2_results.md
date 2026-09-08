# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.0262 | 0.1438 | 0.965 [0.948, 0.979] | +2.0663 [+1.9198, +2.1977] |
| clean | ring_out | 42 | 0.2460 | 0.1510 | 0.429 [0.300, 0.524] | -0.3202 [-0.6474, -0.0230] |
| clean | dense | 4110 | 0.0217 | 0.1449 | 0.979 [0.965, 0.991] | +2.1483 [+1.9706, +2.2946] |
| clean | ring_out_le5 | 120 | 0.1236 | 0.1300 | 0.733 [0.611, 0.817] | +0.6576 [+0.2758, +0.9352] |
| kit_swap | all | 12204 | 2.0241 | 0.1438 | 0.000 [0.000, 0.001] | -2.7715 [-2.8222, -2.7078] |
| kit_swap | ring_out | 42 | 1.5512 | 0.1510 | 0.024 [0.000, 0.062] | -2.6216 [-3.0178, -2.2410] |
| kit_swap | dense | 4110 | 2.0450 | 0.1449 | 0.000 [0.000, 0.000] | -2.7540 [-2.7965, -2.7016] |
| kit_swap | ring_out_le5 | 120 | 1.7988 | 0.1300 | 0.008 [0.000, 0.019] | -2.8027 [-3.0321, -2.6523] |
| full_mask | all | 12204 | 0.0332 | 1.3998 | 0.999 [0.999, 1.000] | +4.2306 [+4.0851, +4.3528] |
| full_mask | ring_out | 42 | 0.3080 | 1.4218 | 0.976 [0.917, 1.000] | +2.1868 [+1.6804, +2.6495] |
| full_mask | dense | 4110 | 0.0265 | 1.3824 | 1.000 [1.000, 1.000] | +4.2665 [+4.0909, +4.4068] |
| full_mask | ring_out_le5 | 120 | 0.1702 | 1.4537 | 0.983 [0.956, 1.000] | +3.0488 [+2.6579, +3.3595] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.999, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.965, log-ratio +2.0663 [+1.9198, +2.1977] -> full better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.429, log-ratio -0.3202 [-0.6474, -0.0230] -> action-only better. Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.999, log-ratio +4.2306 [+4.0851, +4.3528] -> full better; kit-swapped s_t, all: n=12204, win 0.000, log-ratio -2.7715 [-2.8222, -2.7078] -> action-only better. Q1 predicts the full model wins under a full mask.
