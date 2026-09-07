# E1 dynamics sanity — drumjepa_v1_s2 (epoch 19, validation)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.654 [0.620, 0.687] | 0.1169 | 0.1186 |
| state_random | 12204 | 0.979 [0.976, 0.983] | 0.1169 | 0.1425 |
| action_shift | 12204 | 0.982 [0.968, 0.991] | 0.1169 | 0.1571 |
| action_random | 12204 | 0.994 [0.992, 0.997] | 0.1169 | 0.1786 |
| kit_swap | 12204 | 0.994 [0.990, 0.997] | 0.1169 | 0.1471 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.993 |
| 60s Rock | 2034 | 0.989 |
| Studio (Live Room) | 2034 | 0.995 |
| Cassette (Lo-Fi Compress) | 2034 | 0.992 |
| Ele-Drum | 2034 | 0.996 |
| 808 Simple | 2034 | 0.999 |
