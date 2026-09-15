# E1 dynamics sanity — drumjepa_v1_vcrec (epoch 19, validation, train kits)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.548 [0.525, 0.569] | 0.1754 | 0.1760 |
| state_random | 12204 | 0.920 [0.910, 0.928] | 0.1756 | 0.1921 |
| action_shift | 12204 | 0.952 [0.936, 0.966] | 0.1756 | 0.1954 |
| action_random | 12204 | 0.979 [0.974, 0.983] | 0.1756 | 0.2050 |
| kit_swap | 12204 | 0.952 [0.946, 0.957] | 0.1756 | 0.1958 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.964 |
| 60s Rock | 2034 | 0.930 |
| Studio (Live Room) | 2034 | 0.960 |
| Cassette (Lo-Fi Compress) | 2034 | 0.956 |
| Ele-Drum | 2034 | 0.935 |
| 808 Simple | 2034 | 0.968 |
