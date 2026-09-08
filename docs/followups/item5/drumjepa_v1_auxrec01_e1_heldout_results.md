# E1 dynamics sanity — drumjepa_v1_auxrec01 (epoch 19, validation, heldout kits)

16272 transitions, K=4, seed 0.

| perturbation | n | win rate [95% CI] | clean err | perturbed err |
|---|---|---|---|---|
| state_shift | 15840 | 0.526 [0.513, 0.539] | 0.3322 | 0.3507 |
| state_random | 16272 | 0.836 [0.824, 0.847] | 0.3353 | 0.8372 |
| action_shift | 16272 | 0.986 [0.975, 0.995] | 0.3353 | 0.4342 |
| action_random | 16272 | 0.991 [0.987, 0.995] | 0.3353 | 0.4793 |
| kit_swap | 16272 | 0.839 [0.824, 0.852] | 0.3353 | 0.8611 |

kit_swap by the kit of the clean x_t:

| kit | n | win rate |
|---|---|---|
| Acoustic Kit | 2034 | 0.804 |
| Unplugged | 2034 | 0.879 |
| Bigga Bop (Jazz) | 2034 | 0.829 |
| Heavy Metal | 2034 | 0.810 |
| Pop-Rock (Studio) | 2034 | 0.810 |
| Raw Dnb (Layered Hybrid) | 2034 | 0.867 |
| 909 Simple | 2034 | 0.884 |
| Deep Daft | 2034 | 0.826 |
