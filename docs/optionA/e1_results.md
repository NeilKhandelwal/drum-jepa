# E1 dynamics sanity — drumjepa_v1_auxrec (epoch 19, validation)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.557 [0.532, 0.581] | 0.7517 | 0.7531 |
| state_random | 12204 | 0.893 [0.873, 0.914] | 0.7494 | 0.7682 |
| action_shift | 12204 | 0.993 [0.979, 1.000] | 0.7494 | 2.0167 |
| action_random | 12204 | 0.996 [0.990, 1.000] | 0.7494 | 2.0921 |
| kit_swap | 12204 | 0.935 [0.914, 0.955] | 0.7494 | 0.7662 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.905 |
| 60s Rock | 2034 | 0.918 |
| Studio (Live Room) | 2034 | 0.946 |
| Cassette (Lo-Fi Compress) | 2034 | 0.935 |
| Ele-Drum | 2034 | 0.971 |
| 808 Simple | 2034 | 0.937 |
