# E1 dynamics sanity — drumjepa_v1 (epoch 19, validation)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.594 [0.566, 0.620] | 0.0953 | 0.0957 |
| state_random | 12204 | 0.973 [0.968, 0.977] | 0.0954 | 0.1127 |
| action_shift | 12204 | 0.979 [0.966, 0.989] | 0.0954 | 0.1144 |
| action_random | 12204 | 0.993 [0.990, 0.995] | 0.0954 | 0.1191 |
| kit_swap | 12204 | 0.993 [0.987, 0.997] | 0.0954 | 0.1155 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.987 |
| 60s Rock | 2034 | 0.997 |
| Studio (Live Room) | 2034 | 0.992 |
| Cassette (Lo-Fi Compress) | 2034 | 0.990 |
| Ele-Drum | 2034 | 0.998 |
| 808 Simple | 2034 | 0.996 |
