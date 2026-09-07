# E1 dynamics sanity — drumjepa_v1_auxrec01 (epoch 19, validation)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.633 [0.586, 0.680] | 0.0648 | 0.0685 |
| state_random | 12204 | 0.990 [0.987, 0.992] | 0.0673 | 1.4613 |
| action_shift | 12204 | 0.997 [0.994, 1.000] | 0.0673 | 0.1720 |
| action_random | 12204 | 1.000 [1.000, 1.000] | 0.0673 | 0.2196 |
| kit_swap | 12204 | 0.997 [0.995, 0.999] | 0.0673 | 1.7335 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.998 |
| 60s Rock | 2034 | 0.997 |
| Studio (Live Room) | 2034 | 0.994 |
| Cassette (Lo-Fi Compress) | 2034 | 0.998 |
| Ele-Drum | 2034 | 0.998 |
| 808 Simple | 2034 | 0.999 |
