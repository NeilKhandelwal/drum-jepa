# E1 dynamics sanity — drumjepa_v1 (epoch 19, validation, heldout kits)

16272 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 15840 | 0.523 [0.504, 0.542] | 0.2812 | 0.2819 |
| state_random | 16272 | 0.829 [0.813, 0.844] | 0.2812 | 0.3060 |
| action_shift | 16272 | 0.934 [0.911, 0.952] | 0.2812 | 0.2993 |
| action_random | 16272 | 0.941 [0.932, 0.948] | 0.2812 | 0.3025 |
| kit_swap | 16272 | 0.839 [0.820, 0.856] | 0.2812 | 0.3068 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Acoustic Kit | 2034 | 0.823 |
| Unplugged | 2034 | 0.939 |
| Bigga Bop (Jazz) | 2034 | 0.829 |
| Heavy Metal | 2034 | 0.855 |
| Pop-Rock (Studio) | 2034 | 0.779 |
| Raw Dnb (Layered Hybrid) | 2034 | 0.774 |
| 909 Simple | 2034 | 0.881 |
| Deep Daft | 2034 | 0.829 |
