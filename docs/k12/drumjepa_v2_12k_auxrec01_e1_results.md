# E1 dynamics sanity — drumjepa_v2_12k_auxrec01 (epoch 19, validation, train kits)

24408 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 23760 | 0.572 [0.527, 0.618] | 0.0490 | 0.0508 |
| state_random | 24408 | 0.994 [0.991, 0.996] | 0.0557 | 5.2662 |
| action_shift | 24408 | 0.989 [0.976, 0.997] | 0.0557 | 0.1109 |
| action_random | 24408 | 0.992 [0.980, 0.999] | 0.0557 | 0.1636 |
| kit_swap | 24408 | 0.996 [0.991, 0.999] | 0.0557 | 5.7610 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.998 |
| 60s Rock | 2034 | 0.996 |
| Studio (Live Room) | 2034 | 0.994 |
| Cassette (Lo-Fi Compress) | 2034 | 0.995 |
| Ele-Drum | 2034 | 0.997 |
| 808 Simple | 2034 | 0.991 |
| Modern Funk | 2034 | 0.998 |
| Arena Stage | 2034 | 0.998 |
| Tight Prog | 2034 | 0.994 |
| Rockin Gate (80s) | 2034 | 0.995 |
| Super Boom (Layered) | 2034 | 0.995 |
| Nu RNB | 2034 | 0.998 |
