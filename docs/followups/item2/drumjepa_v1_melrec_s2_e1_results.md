# E1 dynamics sanity — drumjepa_v1_melrec_s2 (epoch 19, validation, train kits)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.576 [0.532, 0.623] | 0.0274 | 0.0307 |
| state_random | 12204 | 0.987 [0.983, 0.990] | 0.0293 | 1.7123 |
| action_shift | 12204 | 0.932 [0.897, 0.962] | 0.0293 | 0.0367 |
| action_random | 12204 | 0.957 [0.934, 0.975] | 0.0293 | 0.0694 |
| kit_swap | 12204 | 0.998 [0.996, 1.000] | 0.0293 | 2.0572 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.999 |
| 60s Rock | 2034 | 0.997 |
| Studio (Live Room) | 2034 | 0.998 |
| Cassette (Lo-Fi Compress) | 2034 | 0.998 |
| Ele-Drum | 2034 | 0.998 |
| 808 Simple | 2034 | 0.999 |
