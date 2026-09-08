# E1 dynamics sanity — drumjepa_v1_melrec_s1 (epoch 19, validation, train kits)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.566 [0.529, 0.605] | 0.0385 | 0.0443 |
| state_random | 12204 | 0.984 [0.979, 0.987] | 0.0412 | 1.6917 |
| action_shift | 12204 | 0.918 [0.881, 0.950] | 0.0412 | 0.0485 |
| action_random | 12204 | 0.958 [0.935, 0.975] | 0.0412 | 0.0770 |
| kit_swap | 12204 | 0.998 [0.996, 0.999] | 0.0412 | 2.0067 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.999 |
| 60s Rock | 2034 | 0.999 |
| Studio (Live Room) | 2034 | 0.996 |
| Cassette (Lo-Fi Compress) | 2034 | 0.998 |
| Ele-Drum | 2034 | 0.999 |
| 808 Simple | 2034 | 0.999 |
