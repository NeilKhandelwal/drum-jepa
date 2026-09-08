# E3 kit latent — drumjepa_v1_s1 (epoch 19), 39998 train clips, validation and test in full

Clip embedding = mean over the 128 state tokens of the encoder output. The 14-way task is legitimate because the *training* clips of the held-out kits are embedded too: no kit was ever trained on by the JEPA, only by the probe. Chance is 0.167 (6-way) and 0.071 (14-way).

| task | representation | linear val | linear test | MLP val | MLP test |
|---|---|---|---|---|---|
| 6way | teacher | 0.995 | 0.998 | 0.993 | 0.997 |
| 6way | student | 0.994 | 0.998 | 0.992 | 0.997 |
| 6way | random | 0.919 | 0.949 | 0.919 | 0.952 |
| 6way | raw_mel | 0.995 | 0.997 | 0.986 | 0.995 |
| 14way | teacher | 0.912 | 0.943 | 0.909 | 0.941 |
| 14way | student | 0.912 | 0.943 | 0.908 | 0.942 |
| 14way | random | 0.858 | 0.908 | 0.887 | 0.926 |
| 14way | raw_mel | 0.994 | 0.996 | 0.988 | 0.991 |

Per-kit recall, 14-way linear probe on the teacher embedding:

| kit | split | validation | test |
|---|---|---|---|
| Jazz | train | 0.976 | 0.988 |
| 60s Rock | train | 0.965 | 0.978 |
| Studio (Live Room) | train | 0.967 | 0.981 |
| Cassette (Lo-Fi Compress) | train | 0.969 | 0.991 |
| Ele-Drum | train | 0.976 | 0.984 |
| 808 Simple | train | 0.996 | 0.994 |
| Acoustic Kit | held out | 0.777 | 0.877 |
| Unplugged | held out | 0.931 | 0.949 |
| Bigga Bop (Jazz) | held out | 0.914 | 0.938 |
| Heavy Metal | held out | 0.811 | 0.856 |
| Pop-Rock (Studio) | held out | 0.826 | 0.901 |
| Raw Dnb (Layered Hybrid) | held out | 0.917 | 0.933 |
| 909 Simple | held out | 0.915 | 0.953 |
| Deep Daft | held out | 0.828 | 0.883 |

## Counterfactual geometry (validation)

Ordered kit pairs: 182. Delta consistency on the same 512 windows for every representation, 95% CI from a 1000-resample bootstrap over windows. The random-init encoder and the raw mean log-mel are run through the identical pipeline, so the trained encoder's numbers only count if they beat both.

| representation | within pair, different windows | between disjoint pairs, different windows | between disjoint pairs, pair-mean directions |
|---|---|---|---|
| teacher | 0.356 [0.349, 0.365] | 0.074 [0.072, 0.076] | 0.232 [0.228, 0.233] |
| random | 0.353 [0.340, 0.371] | 0.199 [0.190, 0.211] | 0.548 [0.540, 0.555] |
| raw_mel | 0.500 [0.483, 0.519] | 0.149 [0.143, 0.155] | 0.284 [0.279, 0.289] |

The first two columns are like for like: both average a cosine between two individual window deltas, so both carry the same per-window noise, and their ratio is what says whether a displacement is specific to its kit pair. Both between-pair columns are absolute cosines, because (D,C) is in the pair list whenever (C,D) is and carries the negated delta, so the signed mean is identically zero. The third column averages the noise out first and asks how aligned whole kit vectors are; two unrelated directions give about 0.053 in 229-d, 0.050 in 256-d.

Kit-vector transfer: the mean delta of a pair is estimated on half the windows, added to e_A on the other half, and scored by whether the nearest kit centroid (cosine, centroids from the train split of the same representation) is B.

| representation | all (n=182) | both_train (n=30) | one_heldout (n=96) | both_heldout (n=56) |
|---|---|---|---|---|
| teacher | 0.745 | 0.968 | 0.832 | 0.477 |
| random | 0.112 | 0.146 | 0.116 | 0.085 |
| raw_mel | 0.302 | 0.377 | 0.310 | 0.248 |

Predictor-side swap, target fixed. The target is kit B's x_{t+1} in both arms and only s_t changes, so the two errors are scored against the same teacher target and the comparison stays valid when B is a kit no encoder was trained on. Win rate is P[err with s_t from B < err with s_t from A]; A is always a train kit, B is drawn from the other 13, same K=4 E1 masks. Validation has 2034 windows present on all 14 kits and --max-swaps was 4000. CI: cluster bootstrap over sequences.

| B | n | win rate [95% CI] | err s_t from B | err s_t from A |
|---|---|---|---|---|
| all | 2034 | 0.928 [0.916, 0.939] | 0.1791 | 0.2377 |
| B_train_kit | 783 | 0.994 [0.987, 0.999] | 0.0778 | 0.1286 |
| B_heldout_kit | 1251 | 0.886 [0.869, 0.903] | 0.2425 | 0.3060 |

The reverse reading, holding s_t at kit B and swapping the target, is reported for train-kit B only. With a held-out B the two targets are not exchangeable: the teacher never saw that kit, so the error against its target is inflated whatever the prediction, and the comparison measures target familiarity rather than kit tracking.

| B | n | win rate [95% CI] | err vs target B | err vs target A |
|---|---|---|---|---|
| B_train_kit | 783 | 0.923 [0.899, 0.945] | 0.0778 | 0.1280 |

## Held-out kits

Where the 6-way linear probe (teacher, trained on the 6 train kits only) sends each held-out kit's validation clips. Rows sum to 1; the expected column from configs/kits_v1.yaml is marked *.

| held-out kit | Jazz | 60s Rock | Studio (Live Room) | Cassette (Lo-Fi Compress) | Ele-Drum | 808 Simple |
|---|---|---|---|---|---|---|
| Acoustic Kit | 0.12* | 0.09* | 0.71* | 0.02* | 0.04 | 0.02 |
| Unplugged | 0.85* | 0.06* | 0.06* | 0.01* | 0.02 | 0.00 |
| Bigga Bop (Jazz) | 0.16* | 0.29 | 0.20 | 0.06 | 0.30 | 0.00 |
| Heavy Metal | 0.39* | 0.24* | 0.21* | 0.03* | 0.12 | 0.02 |
| Pop-Rock (Studio) | 0.01 | 0.34 | 0.35* | 0.22 | 0.07 | 0.00 |
| Raw Dnb (Layered Hybrid) | 0.01 | 0.86 | 0.05 | 0.02 | 0.05* | 0.01* |
| 909 Simple | 0.21 | 0.35 | 0.04 | 0.06 | 0.09 | 0.25* |
| Deep Daft | 0.01 | 0.48 | 0.20 | 0.02 | 0.28* | 0.00* |

Nearest other kit by cosine distance between validation centroids:

| kit | nearest | cosine distance |
|---|---|---|
| Jazz | Unplugged | 0.0759 |
| 60s Rock | Raw Dnb (Layered Hybrid) | 0.0364 |
| Studio (Live Room) | Acoustic Kit | 0.0587 |
| Cassette (Lo-Fi Compress) | Pop-Rock (Studio) | 0.2189 |
| Ele-Drum | Deep Daft | 0.2470 |
| 808 Simple | 909 Simple | 0.1938 |
| Acoustic Kit | Pop-Rock (Studio) | 0.0403 |
| Unplugged | Heavy Metal | 0.0365 |
| Bigga Bop (Jazz) | Deep Daft | 0.0120 |
| Heavy Metal | Bigga Bop (Jazz) | 0.0151 |
| Pop-Rock (Studio) | Deep Daft | 0.0248 |
| Raw Dnb (Layered Hybrid) | 60s Rock | 0.0364 |
| 909 Simple | Heavy Metal | 0.0229 |
| Deep Daft | Bigga Bop (Jazz) | 0.0120 |

## Q2

Kit identity is readable from s_t: the 14-way linear probe on the teacher embedding scores 0.943 on test against 0.071 chance, and the 6-way probe on the train kits scores 0.998 against 0.167.
But notes/decisions.md sets the bar at beating both baselines, and on the 14-way task the teacher beats the random-init encoder: random-init encoder 0.908, raw mean log-mel 0.996. Kit identity is present in the embedding but no more linearly accessible there than in the mel spectrum it was computed from, so the probe alone does not show the model built a kit latent.
The counterfactual displacement e_B - e_A points the same way across performances (teacher: within-pair cosine 0.356 [0.349, 0.365] against 0.074 between disjoint kit pairs, a ratio of 4.8) and a kit vector estimated on half the windows moves a clip to the right kit's centroid on the other half 0.745 of the time (0.968 between train kits, 0.477 between two held-out kits).
Run through the same pipeline the controls give within/between ratios of 1.8 (random), 3.4 (raw_mel) against the teacher's 4.8, and transfer 0.112 (random), 0.302 (raw_mel) against 0.745 — both below the teacher, so this part of the geometry is built by training and not inherited from the spectrum; raw_mel has a higher raw within-pair cosine than the teacher but spread it over unrelated kit pairs too, which is what the ratio and the transfer test catch.
With the target held fixed at kit B's x_(t+1) and only s_t varying, giving the predictor the matching kit's state beats giving it a train kit's state on 0.994 of windows [0.987, 0.999] when B is a train kit and 0.886 [0.869, 0.903] when B was never trained on, so the predictor uses the kit of s_t, and it does so for unseen kits too.
Held-out kits embed sensibly on the whole: 4 of 8 send most of their clips to the train kit the annotations predict (Bigga Bop (Jazz), Raw Dnb (Layered Hybrid), 909 Simple, Deep Daft did not), and the lowest 14-way test recall among them is 0.856 against 0.071 chance, so the encoder places unseen kits in distinct, mostly plausible regions.
