# E1 dynamics sanity — drumjepa_v1_vcrec_s2 (epoch 19, validation, train kits)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.535 [0.511, 0.557] | 0.2229 | 0.2232 |
| state_random | 12204 | 0.892 [0.883, 0.901] | 0.2229 | 0.2417 |
| action_shift | 12204 | 0.811 [0.787, 0.835] | 0.2229 | 0.2284 |
| action_random | 12204 | 0.884 [0.873, 0.895] | 0.2229 | 0.2319 |
| kit_swap | 12204 | 0.940 [0.922, 0.952] | 0.2229 | 0.2452 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.951 |
| 60s Rock | 2034 | 0.903 |
| Studio (Live Room) | 2034 | 0.930 |
| Cassette (Lo-Fi Compress) | 2034 | 0.976 |
| Ele-Drum | 2034 | 0.934 |
| 808 Simple | 2034 | 0.946 |
