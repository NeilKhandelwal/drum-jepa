# E1 dynamics sanity — drumjepa_v2_12k (epoch 19, validation, train kits)

24408 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 23760 | 0.547 [0.487, 0.601] | 0.0163 | 0.0181 |
| state_random | 24408 | 0.991 [0.987, 0.993] | 0.0189 | 2.6978 |
| action_shift | 24408 | 0.951 [0.925, 0.971] | 0.0189 | 0.0219 |
| action_random | 24408 | 0.951 [0.927, 0.967] | 0.0189 | 0.0225 |
| kit_swap | 24408 | 0.996 [0.994, 0.998] | 0.0189 | 2.9624 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.999 |
| 60s Rock | 2034 | 0.996 |
| Studio (Live Room) | 2034 | 0.998 |
| Cassette (Lo-Fi Compress) | 2034 | 0.995 |
| Ele-Drum | 2034 | 0.995 |
| 808 Simple | 2034 | 0.997 |
| Modern Funk | 2034 | 0.996 |
| Arena Stage | 2034 | 0.997 |
| Tight Prog | 2034 | 0.994 |
| Rockin Gate (80s) | 2034 | 0.998 |
| Super Boom (Layered) | 2034 | 0.998 |
| Nu RNB | 2034 | 0.997 |
