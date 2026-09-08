# E1 dynamics sanity — drumjepa_v1_e40 (epoch 19, validation, train kits)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.582 [0.557, 0.607] | 0.2182 | 0.2193 |
| state_random | 12204 | 0.976 [0.972, 0.980] | 0.2182 | 0.3846 |
| action_shift | 12204 | 0.978 [0.963, 0.989] | 0.2182 | 0.3172 |
| action_random | 12204 | 0.993 [0.988, 0.996] | 0.2182 | 0.3416 |
| kit_swap | 12204 | 0.995 [0.989, 0.998] | 0.2182 | 0.4236 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.997 |
| 60s Rock | 2034 | 0.995 |
| Studio (Live Room) | 2034 | 0.995 |
| Cassette (Lo-Fi Compress) | 2034 | 0.992 |
| Ele-Drum | 2034 | 0.994 |
| 808 Simple | 2034 | 0.996 |
