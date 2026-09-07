# E1 dynamics sanity — drumjepa_v1_auxrec01_s2 (epoch 19, validation)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.581 [0.543, 0.617] | 0.0585 | 0.0663 |
| state_random | 12204 | 0.990 [0.987, 0.993] | 0.0622 | 2.5404 |
| action_shift | 12204 | 0.987 [0.973, 0.997] | 0.0622 | 0.1070 |
| action_random | 12204 | 0.998 [0.996, 0.999] | 0.0622 | 0.1825 |
| kit_swap | 12204 | 0.997 [0.995, 0.999] | 0.0622 | 3.0125 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.997 |
| 60s Rock | 2034 | 0.998 |
| Studio (Live Room) | 2034 | 0.996 |
| Cassette (Lo-Fi Compress) | 2034 | 0.999 |
| Ele-Drum | 2034 | 0.996 |
| 808 Simple | 2034 | 0.999 |
