# E1 dynamics sanity — drumjepa_v1_vcrec_s1 (epoch 19, validation, train kits)

12204 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 11880 | 0.508 [0.486, 0.530] | 0.1724 | 0.1723 |
| state_random | 12204 | 0.812 [0.799, 0.824] | 0.1724 | 0.1763 |
| action_shift | 12204 | 0.680 [0.662, 0.700] | 0.1724 | 0.1745 |
| action_random | 12204 | 0.800 [0.784, 0.814] | 0.1724 | 0.1767 |
| kit_swap | 12204 | 0.852 [0.838, 0.865] | 0.1724 | 0.1769 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Jazz | 2034 | 0.930 |
| 60s Rock | 2034 | 0.793 |
| Studio (Live Room) | 2034 | 0.839 |
| Cassette (Lo-Fi Compress) | 2034 | 0.889 |
| Ele-Drum | 2034 | 0.849 |
| 808 Simple | 2034 | 0.812 |
