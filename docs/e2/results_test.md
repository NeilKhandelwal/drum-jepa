# E2 state ablation — full (epoch 19) vs action-only (epoch 19), test

11760 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 11760 | 0.0915 | 0.1389 | 0.769 [0.734, 0.798] | +0.3149 [+0.2709, +0.3539] |
| clean | ring_out | 36 | 0.1649 | 0.2882 | 0.667 [0.333, 0.733] | +0.4620 [-0.3945, +0.6334] |
| clean | dense | 3456 | 0.0947 | 0.1560 | 0.845 [0.776, 0.881] | +0.4166 [+0.3054, +0.4867] |
| clean | ring_out_le5 | 48 | 0.1602 | 0.2781 | 0.667 [0.333, 0.714] | +0.4364 [-0.3945, +0.5551] |
| kit_swap | all | 11760 | 0.1101 | 0.1389 | 0.610 [0.566, 0.646] | +0.1308 [+0.0877, +0.1679] |
| kit_swap | ring_out | 36 | 0.1995 | 0.2882 | 0.500 [0.333, 0.533] | +0.1948 [-0.5087, +0.3355] |
| kit_swap | dense | 3456 | 0.1160 | 0.1560 | 0.699 [0.609, 0.744] | +0.2119 [+0.1099, +0.2647] |
| kit_swap | ring_out_le5 | 48 | 0.1921 | 0.2781 | 0.521 [0.333, 0.548] | +0.1984 [-0.5087, +0.2995] |
| full_mask | all | 11760 | 0.6537 | 1.4376 | 1.000 [1.000, 1.000] | +0.8139 [+0.7587, +0.8650] |
| full_mask | ring_out | 36 | 0.6474 | 1.5424 | 1.000 [1.000, 1.000] | +0.9296 [+0.8394, +0.9477] |
| full_mask | dense | 3456 | 0.5968 | 1.4176 | 1.000 [1.000, 1.000] | +0.9031 [+0.7467, +1.0269] |
| full_mask | ring_out_le5 | 48 | 0.6317 | 1.4787 | 1.000 [1.000, 1.000] | +0.9048 [+0.8394, +0.9141] |

Own clean-vs-kit-swapped win rate (the E1 number, n=11760): full 0.995, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 36 transitions, so the <= 5 onset subset `ring_out_le5` (48) is reported as well.

## Q1

1. Clean, all: n=11760, win 0.769, log-ratio +0.3149 [+0.2709, +0.3539] -> full better. Q1 predicts a near tie.
2. Clean, ring_out: n=36, win 0.667, log-ratio +0.4620 [-0.3945, +0.6334] -> tie (CI spans 0). Q1 predicts the full model wins.
3. Full mask, all: n=11760, win 1.000, log-ratio +0.8139 [+0.7587, +0.8650] -> full better; kit-swapped s_t, all: n=11760, win 0.610, log-ratio +0.1308 [+0.0877, +0.1679] -> full better. Q1 predicts the full model wins under a full mask.
