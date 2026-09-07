# E1 dynamics sanity — drumjepa_v1_auxrec03 (epoch 19, validation)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.682 [0.641, 0.723] | 0.4478 | 0.4529 |
| state_random | 12204 | 0.993 [0.991, 0.995] | 0.4471 | 0.5266 |
| action_shift | 12204 | 0.998 [0.994, 1.000] | 0.4471 | 1.2297 |
| action_random | 12204 | 1.000 [0.999, 1.000] | 0.4471 | 1.3550 |
| kit_swap | 12204 | 0.997 [0.995, 0.999] | 0.4471 | 0.5449 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.998 |
| 60s Rock | 2034 | 0.998 |
| Studio (Live Room) | 2034 | 0.994 |
| Cassette (Lo-Fi Compress) | 2034 | 0.998 |
| Ele-Drum | 2034 | 0.999 |
| 808 Simple | 2034 | 0.996 |
