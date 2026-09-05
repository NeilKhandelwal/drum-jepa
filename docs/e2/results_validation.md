# E2 state ablation — full (epoch 19) vs action-only (epoch 19), validation

12204 transitions, K=4 (full_mask: K=1), seed 0. Win rate is P[err full < err action-only]; log-ratio is mean log(err action-only / err full), positive = full better. CIs: 1000-resample cluster bootstrap over sequences.

| condition | subset | n | err full | err action-only | win rate [95% CI] | log-ratio [95% CI] |
|---|---|---|---|---|---|---|
| clean | all | 12204 | 0.0954 | 0.1438 | 0.780 [0.756, 0.802] | +0.3196 [+0.2904, +0.3523] |
| clean | ring_out | 42 | 0.0969 | 0.1510 | 0.595 [0.467, 0.690] | +0.2088 [-0.0261, +0.3527] |
| clean | dense | 4110 | 0.0942 | 0.1449 | 0.820 [0.786, 0.848] | +0.3574 [+0.3176, +0.4017] |
| clean | ring_out_le5 | 120 | 0.0925 | 0.1300 | 0.650 [0.524, 0.718] | +0.2101 [+0.0649, +0.2792] |
| kit_swap | all | 12204 | 0.1155 | 0.1438 | 0.615 [0.583, 0.643] | +0.1288 [+0.0954, +0.1622] |
| kit_swap | ring_out | 42 | 0.1319 | 0.1510 | 0.452 [0.333, 0.556] | -0.1225 [-0.2669, -0.0423] |
| kit_swap | dense | 4110 | 0.1158 | 0.1449 | 0.640 [0.590, 0.690] | +0.1523 [+0.1008, +0.2063] |
| kit_swap | ring_out_le5 | 120 | 0.1205 | 0.1300 | 0.475 [0.350, 0.524] | -0.0490 [-0.1939, +0.0094] |
| full_mask | all | 12204 | 0.6240 | 1.3998 | 1.000 [1.000, 1.000] | +0.8402 [+0.7772, +0.9119] |
| full_mask | ring_out | 42 | 0.4419 | 1.4218 | 1.000 [1.000, 1.000] | +1.2321 [+1.0570, +1.4518] |
| full_mask | dense | 4110 | 0.5949 | 1.3824 | 1.000 [1.000, 1.000] | +0.8790 [+0.7649, +1.0050] |
| full_mask | ring_out_le5 | 120 | 0.5637 | 1.4537 | 1.000 [1.000, 1.000] | +1.0001 [+0.8955, +1.1596] |

Own clean-vs-kit-swapped win rate (the E1 number, n=12204): full 0.994, action-only 0.500 (invariant by construction, so every pair ties and ties count as half; max |clean - swapped| per transition 0.00e+00).

ring_out (<= 2 onsets in the t+1 window) has only 42 transitions, so the <= 5 onset subset `ring_out_le5` (120) is reported as well.

## Q1

1. Clean, all: n=12204, win 0.780, log-ratio +0.3196 [+0.2904, +0.3523] -> full better. Q1 predicts a near tie.
2. Clean, ring_out: n=42, win 0.595, log-ratio +0.2088 [-0.0261, +0.3527] -> tie (CI spans 0). Q1 predicts the full model wins.
3. Full mask, all: n=12204, win 1.000, log-ratio +0.8402 [+0.7772, +0.9119] -> full better; kit-swapped s_t, all: n=12204, win 0.615, log-ratio +0.1288 [+0.0954, +0.1622] -> full better. Q1 predicts the full model wins under a full mask.
