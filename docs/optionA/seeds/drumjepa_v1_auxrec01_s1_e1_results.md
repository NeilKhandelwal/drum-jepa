# E1 dynamics sanity — drumjepa_v1_auxrec01_s1 (epoch 19, validation)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.587 [0.533, 0.644] | 0.0466 | 0.0459 |
| state_random | 12204 | 0.991 [0.987, 0.993] | 0.0497 | 2.5209 |
| action_shift | 12204 | 0.985 [0.966, 0.997] | 0.0497 | 0.0889 |
| action_random | 12204 | 0.998 [0.996, 0.999] | 0.0497 | 0.1260 |
| kit_swap | 12204 | 0.995 [0.988, 0.999] | 0.0497 | 3.0021 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.998 |
| 60s Rock | 2034 | 0.996 |
| Studio (Live Room) | 2034 | 0.993 |
| Cassette (Lo-Fi Compress) | 2034 | 0.998 |
| Ele-Drum | 2034 | 0.990 |
| 808 Simple | 2034 | 0.994 |
