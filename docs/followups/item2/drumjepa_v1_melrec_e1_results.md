# E1 dynamics sanity — drumjepa_v1_melrec (epoch 19, validation, train kits)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.574 [0.538, 0.610] | 0.0244 | 0.0277 |
| state_random | 12204 | 0.985 [0.981, 0.989] | 0.0262 | 1.6947 |
| action_shift | 12204 | 0.931 [0.900, 0.958] | 0.0262 | 0.0351 |
| action_random | 12204 | 0.963 [0.946, 0.977] | 0.0262 | 0.0689 |
| kit_swap | 12204 | 0.998 [0.996, 1.000] | 0.0262 | 2.0241 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.999 |
| 60s Rock | 2034 | 0.997 |
| Studio (Live Room) | 2034 | 0.998 |
| Cassette (Lo-Fi Compress) | 2034 | 0.998 |
| Ele-Drum | 2034 | 0.999 |
| 808 Simple | 2034 | 0.999 |
