# E3 kit latent — drumjepa_v1_auxrec01 (epoch 19), 39998 train clips, validation and test in full

Clip embedding = mean over the 128 state tokens of the encoder output. The 14-way task is legitimate because the *training* clips of the held-out kits are embedded too: no kit was ever trained on by the JEPA, only by the probe. Chance is 0.167 (6-way) and 0.071 (14-way).

| task | representation | linear val | linear test | MLP val | MLP test |
|---|---|---|---|---|---|
| 6way | teacher | 0.996 | 0.997 | 0.996 | 0.997 |
| 6way | student | 0.996 | 0.997 | 0.995 | 0.997 |
| 6way | random | 0.919 | 0.949 | 0.919 | 0.952 |
| 6way | raw_mel | 0.995 | 0.997 | 0.986 | 0.995 |
| 14way | teacher | 0.975 | 0.986 | 0.977 | 0.988 |
| 14way | student | 0.975 | 0.986 | 0.976 | 0.986 |
| 14way | random | 0.858 | 0.908 | 0.887 | 0.926 |
| 14way | raw_mel | 0.994 | 0.996 | 0.988 | 0.991 |

Per-kit recall, 14-way linear probe on the teacher embedding:

| kit | split | validation | test |
|---|---|---|---|
| Jazz | train | 0.989 | 0.992 |
| 60s Rock | train | 0.973 | 0.984 |
| Studio (Live Room) | train | 0.991 | 0.989 |
| Cassette (Lo-Fi Compress) | train | 0.990 | 0.995 |
| Ele-Drum | train | 0.987 | 0.994 |
| 808 Simple | train | 0.997 | 0.996 |
| Acoustic Kit | held out | 0.945 | 0.969 |
| Unplugged | held out | 0.984 | 0.989 |
| Bigga Bop (Jazz) | held out | 0.958 | 0.989 |
| Heavy Metal | held out | 0.952 | 0.970 |
| Pop-Rock (Studio) | held out | 0.952 | 0.974 |
| Raw Dnb (Layered Hybrid) | held out | 0.965 | 0.990 |
| 909 Simple | held out | 0.980 | 0.991 |
| Deep Daft | held out | 0.984 | 0.978 |

## Counterfactual geometry (validation)

Ordered kit pairs: 182. Delta consistency on the same 512 windows for every representation, 95% CI from a 1000-resample bootstrap over windows. The random-init encoder and the raw mean log-mel are run through the identical pipeline, so the trained encoder's numbers only count if they beat both.

| representation | within pair, different windows | between disjoint pairs, different windows | between disjoint pairs, pair-mean directions |
|---|---|---|---|
| teacher | 0.637 [0.627, 0.648] | 0.129 [0.126, 0.133] | 0.215 [0.213, 0.218] |
| random | 0.353 [0.340, 0.371] | 0.199 [0.190, 0.211] | 0.548 [0.540, 0.555] |
| raw_mel | 0.500 [0.483, 0.519] | 0.149 [0.143, 0.155] | 0.284 [0.279, 0.289] |

The first two columns are like for like: both average a cosine between two individual window deltas, so both carry the same per-window noise, and their ratio is what says whether a displacement is specific to its kit pair. Both between-pair columns are absolute cosines, because (D,C) is in the pair list whenever (C,D) is and carries the negated delta, so the signed mean is identically zero. The third column averages the noise out first and asks how aligned whole kit vectors are; two unrelated directions give about 0.053 in 229-d, 0.050 in 256-d.

Kit-vector transfer: the mean delta of a pair is estimated on half the windows, added to e_A on the other half, and scored by whether the nearest kit centroid (cosine, centroids from the train split of the same representation) is B.

| representation | all (n=182) | both_train (n=30) | one_heldout (n=96) | both_heldout (n=56) |
|---|---|---|---|---|
| teacher | 0.761 | 0.954 | 0.825 | 0.547 |
| random | 0.112 | 0.146 | 0.116 | 0.085 |
| raw_mel | 0.302 | 0.377 | 0.310 | 0.248 |

Predictor-side swap, target fixed. The target is kit B's x_{t+1} in both arms and only s_t changes, so the two errors are scored against the same teacher target and the comparison stays valid when B is a kit no encoder was trained on. Win rate is P[err with s_t from B < err with s_t from A]; A is always a train kit, B is drawn from the other 13, same K=4 E1 masks. Validation has 2034 windows present on all 14 kits and --max-swaps was 4000. CI: cluster bootstrap over sequences.

| B | n | win rate [95% CI] | err s_t from B | err s_t from A |
|---|---|---|---|---|
| all | 2034 | 0.971 [0.963, 0.979] | 0.2348 | 1.5044 |
| B_train_kit | 783 | 0.995 [0.989, 0.999] | 0.0712 | 1.7317 |
| B_heldout_kit | 1251 | 0.955 [0.943, 0.967] | 0.3371 | 1.3622 |

The reverse reading, holding s_t at kit B and swapping the target, is reported for train-kit B only. With a held-out B the two targets are not exchangeable: the teacher never saw that kit, so the error against its target is inflated whatever the prediction, and the comparison measures target familiarity rather than kit tracking.

| B | n | win rate [95% CI] | err vs target B | err vs target A |
|---|---|---|---|---|
| B_train_kit | 783 | 0.996 [0.991, 1.000] | 0.0712 | 1.7327 |

## Held-out kits

Where the 6-way linear probe (teacher, trained on the 6 train kits only) sends each held-out kit's validation clips. Rows sum to 1; the expected column from configs/kits_v1.yaml is marked *.

| held-out kit | Jazz | 60s Rock | Studio (Live Room) | Cassette (Lo-Fi Compress) | Ele-Drum | 808 Simple |
|---|---|---|---|---|---|---|
| Acoustic Kit | 0.39* | 0.20* | 0.37* | 0.00* | 0.03 | 0.00 |
| Unplugged | 0.93* | 0.03* | 0.01* | 0.02* | 0.01 | 0.00 |
| Bigga Bop (Jazz) | 0.45* | 0.10 | 0.04 | 0.04 | 0.36 | 0.00 |
| Heavy Metal | 0.64* | 0.31* | 0.00* | 0.01* | 0.03 | 0.01 |
| Pop-Rock (Studio) | 0.06 | 0.68 | 0.05* | 0.08 | 0.13 | 0.00 |
| Raw Dnb (Layered Hybrid) | 0.17 | 0.72 | 0.03 | 0.01 | 0.05* | 0.02* |
| 909 Simple | 0.49 | 0.15 | 0.00 | 0.00 | 0.03 | 0.33* |
| Deep Daft | 0.10 | 0.71 | 0.03 | 0.05 | 0.09* | 0.00* |

Nearest other kit by cosine distance between validation centroids:

| kit | nearest | cosine distance |
|---|---|---|
| Jazz | Unplugged | 0.0478 |
| 60s Rock | Pop-Rock (Studio) | 0.0386 |
| Studio (Live Room) | Acoustic Kit | 0.1250 |
| Cassette (Lo-Fi Compress) | Pop-Rock (Studio) | 0.2501 |
| Ele-Drum | Bigga Bop (Jazz) | 0.1929 |
| 808 Simple | 909 Simple | 0.1567 |
| Acoustic Kit | Deep Daft | 0.0380 |
| Unplugged | Jazz | 0.0478 |
| Bigga Bop (Jazz) | Heavy Metal | 0.0441 |
| Heavy Metal | Acoustic Kit | 0.0405 |
| Pop-Rock (Studio) | Deep Daft | 0.0084 |
| Raw Dnb (Layered Hybrid) | Heavy Metal | 0.0405 |
| 909 Simple | Unplugged | 0.0571 |
| Deep Daft | Pop-Rock (Studio) | 0.0084 |

## Q2

Kit identity is readable from s_t: the 14-way linear probe on the teacher embedding scores 0.986 on test against 0.071 chance, and the 6-way probe on the train kits scores 0.997 against 0.167.
But notes/decisions.md sets the bar at beating both baselines, and on the 14-way task the teacher beats the random-init encoder: random-init encoder 0.908, raw mean log-mel 0.996. Kit identity is present in the embedding but no more linearly accessible there than in the mel spectrum it was computed from, so the probe alone does not show the model built a kit latent.
The counterfactual displacement e_B - e_A points the same way across performances (teacher: within-pair cosine 0.637 [0.627, 0.648] against 0.129 between disjoint kit pairs, a ratio of 4.9) and a kit vector estimated on half the windows moves a clip to the right kit's centroid on the other half 0.761 of the time (0.954 between train kits, 0.547 between two held-out kits).
Run through the same pipeline the controls give within/between ratios of 1.8 (random), 3.4 (raw_mel) against the teacher's 4.9, and transfer 0.112 (random), 0.302 (raw_mel) against 0.761 — both below the teacher, so this part of the geometry is built by training and not inherited from the spectrum.
With the target held fixed at kit B's x_(t+1) and only s_t varying, giving the predictor the matching kit's state beats giving it a train kit's state on 0.995 of windows [0.989, 0.999] when B is a train kit and 0.955 [0.943, 0.967] when B was never trained on, so the predictor uses the kit of s_t, and it does so for unseen kits too.
Held-out kits embed sensibly on the whole: 4 of 8 send most of their clips to the train kit the annotations predict (Pop-Rock (Studio), Raw Dnb (Layered Hybrid), 909 Simple, Deep Daft did not), and the lowest 14-way test recall among them is 0.969 against 0.071 chance, so the encoder places unseen kits in distinct, mostly plausible regions.
