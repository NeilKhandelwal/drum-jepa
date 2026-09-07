# E1 dynamics sanity — drumjepa_v1_s1 (epoch 19, validation)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.634 [0.602, 0.666] | 0.0767 | 0.0772 |
| state_random | 12204 | 0.980 [0.975, 0.983] | 0.0767 | 0.1188 |
| action_shift | 12204 | 0.983 [0.970, 0.992] | 0.0767 | 0.0980 |
| action_random | 12204 | 0.994 [0.990, 0.997] | 0.0767 | 0.1049 |
| kit_swap | 12204 | 0.995 [0.989, 0.999] | 0.0767 | 0.1275 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.998 |
| 60s Rock | 2034 | 0.993 |
| Studio (Live Room) | 2034 | 0.995 |
| Cassette (Lo-Fi Compress) | 2034 | 0.994 |
| Ele-Drum | 2034 | 0.994 |
| 808 Simple | 2034 | 0.998 |
