# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.0412 | 0.1438 | 0.928 [0.897, 0.953] | +1.8600 [+1.6758, +2.0143] |
| clean | ring_out | 42 | 0.2678 | 0.1510 | 0.452 [0.278, 0.595] | -0.2360 [-0.7694, +0.2074] |
| clean | dense | 4110 | 0.0430 | 0.1449 | 0.917 [0.856, 0.970] | +1.8125 [+1.5338, +2.0654] |
| clean | ring_out_le5 | 120 | 0.1556 | 0.1300 | 0.675 [0.529, 0.770] | +0.4623 [+0.0713, +0.6793] |
| kit_swap | all | 12204 | 2.0067 | 0.1438 | 0.000 [0.000, 0.001] | -2.7665 [-2.8139, -2.7097] |
| kit_swap | ring_out | 42 | 1.5955 | 0.1510 | 0.024 [0.000, 0.062] | -2.6841 [-3.0667, -2.3737] |
| kit_swap | dense | 4110 | 2.0295 | 0.1449 | 0.000 [0.000, 0.000] | -2.7475 [-2.7900, -2.6909] |
| kit_swap | ring_out_le5 | 120 | 1.8410 | 0.1300 | 0.008 [0.000, 0.019] | -2.8550 [-3.0248, -2.7755] |
| full_mask | all | 12204 | 0.0467 | 1.3998 | 0.999 [0.998, 1.000] | +4.0851 [+3.8991, +4.2337] |
| full_mask | ring_out | 42 | 0.3073 | 1.4218 | 0.952 [0.900, 1.000] | +2.2816 [+1.6963, +2.8580] |
| full_mask | dense | 4110 | 0.0471 | 1.3824 | 1.000 [0.999, 1.000] | +4.0109 [+3.7308, +4.2677] |
| full_mask | ring_out_le5 | 120 | 0.1797 | 1.4537 | 0.975 [0.952, 1.000] | +2.9380 [+2.6251, +3.1452] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.998, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.928, log-ratio +1.8600 [+1.6758, +2.0143] -> full better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.452, log-ratio -0.2360 [-0.7694, +0.2074] -> tie (CI spans 0). Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 0.999, log-ratio +4.0851 [+3.8991, +4.2337] -> full better; kit-swapped s_t, all: n=12204, win 0.000, log-ratio -2.7665 [-2.8139, -2.7097] -> action-only better. Q1 predicts the full model wins under a full mask.
