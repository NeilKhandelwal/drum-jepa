# E1 dynamics sanity — drumjepa_v1_e40 (epoch 39, validation, train kits)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.546 [0.517, 0.574] | 0.0972 | 0.0974 |
| state_random | 12204 | 0.946 [0.942, 0.951] | 0.0972 | 0.2332 |
| action_shift | 12204 | 0.961 [0.944, 0.978] | 0.0972 | 0.1537 |
| action_random | 12204 | 0.984 [0.979, 0.989] | 0.0972 | 0.1833 |
| kit_swap | 12204 | 0.988 [0.985, 0.991] | 0.0972 | 0.2648 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.993 |
| 60s Rock | 2034 | 0.995 |
| Studio (Live Room) | 2034 | 0.991 |
| Cassette (Lo-Fi Compress) | 2034 | 0.974 |
| Ele-Drum | 2034 | 0.996 |
| 808 Simple | 2034 | 0.981 |
