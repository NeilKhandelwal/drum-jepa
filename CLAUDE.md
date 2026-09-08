# drum-jepa — project context for Claude Code

## Who and why
Neil Khandelwal, NYU sophomore (Math + CS), researcher in Yann LeCun's lab under
Dr. Ziyu Wang (day-to-day PI). Ziyu is first author of Music-JEPA (arXiv 2607.22000,
July 2026). He advised: get hands-on JEPA training experience by retraining a small
non-music JEPA, then build an action-conditioned drum JEPA. The Music-JEPA code is
NOT released; we reimplement from the paper. Do not wait for it.

Ziyu already tried SIGReg in Music-JEPA and it did not work well with that
architecture. Music-JEPA uses EMA + stop-grad + LayerNorm for anti-collapse.

Neil is not messaging Ziyu again until there is a result of substance
(target: E1 passed, one figure).

## The project in one paragraph
Action-conditioned JEPA world model of drum sound on E-GMD, following the
Music-JEPA design exactly. State x_t = 2-second log-mel spectrogram of drum
audio. Action y_t = drumroll (instrument classes x time, velocity in cells) plus
the hi-hat pedal CC4 track. Model: state encoder E_s, action encoder E_a,
state predictor f(s_t, a_{t+1}) -> s_{t+1}, action predictor g(a_t) -> a_{t+1}.
Loss = ||f(s_t,a_{t+1}) - sg(s_{t+1})||^2 + 0.5 * ||g(a_t) - sg(a_{t+1})||^2,
teacher encoders via EMA (tau=0.95), LayerNorm on all encoder outputs.

## Music-JEPA spec to port (from the paper)
- Audio: 229-bin log-mel, 10 ms frames, n_fft 2048, hop 160, 2 s segments ->
  x_t in R^{200x229}. Match ZZWaang/audio2midi conventions (third_party/).
- State encoder: ViT, patch 25x15 on the spectrogram -> 16x16=256 tokens, 12 layers,
  d_model 256, d_ff 512, 4 heads.
- Action encoder: ViT on the drumroll, 8 layers, same dims. Paper patches the
  88-pitch pianoroll with 25x6; for our K=14 drumroll patch 25 frames x all 14
  rows (one patch per time step) as the default.
- Hi-hat CC4 track: 1-D conv over 16 time segments, broadcast across pitch/class
  patches, fused with the drumroll embeddings (same as the paper's pedal signal).
- State predictor f: 6-layer transformer over [s_t tokens ; masked s_{t+1} tokens],
  cross-attention to a_{t+1} tokens at every layer (action tokens as K/V only).
  Masking scheme unspecified in the paper -> I-JEPA-style multi-block, 75% to start.
- Action predictor g: 6-layer transformer, no cross-attention, takes a_t and masked
  a_{t+1}, reconstructs a_{t+1}.
- Training: Adam lr 6e-4, wd 1e-4, batch 128, 15-25 epochs, lambda=0.5, tau=0.95.
- Total ~19M params.

Unspecified details (decide, document, don't agonize): mask ratio/block scheme,
K/V-only cross-attention, velocity scaling in drumroll cells, mel normalization.

## Scaffolding sources (all in third_party/, gitignored)
- third_party/ijepa   -> masked-target predictor, EMA target encoder, multiblock masks
                        (read src/train.py and src/masks/multiblock.py first)
- third_party/le-wm   -> action-conditioned predictor pattern (LeWorldModel)
- third_party/audio2midi -> Ziyu's own mel/MIDI preprocessing conventions; match them
- third_party/lejepa  -> SIGReg reference (for E6 later)

## Dataset: E-GMD
- 444 audio hours = 1,059 unique performances x 43 kits (each performance
  re-recorded on every kit). Only ~10 h of unique actions. Roland TD-17.
- Official splits (train/val/test) are by sequence and already in the CSV `split`
  column. NEVER split by (sequence, kit) — identical actions across kits leak.
- We subset by kit: 6 train kits + 8 held-out kits, all sequences on each.
  Train kits chosen: 808 Simple, Jazz, Ele-Drum, Cassette, Studio, 60s Rock.
  Held-out kits: TBD (8, spread acoustic->electronic). Exact kit_name strings
  from the CSV go in configs/kits_v1.yaml.
- Every performance has 42 counterfactuals (same action, different kit). This
  is the core experimental lever. Caveat found 2026-09-04: the shipped MIDI is
  NOT identical across kits; 18/43 kits remap pads (hi-hat -> 54, tom rim -> 56
  or 39, x-stick -> rim). Hit counts are conserved, so build the action from the
  canonical MIDI (configs/kits_v1.yaml: canonical_kit), never from the kit's
  own file. Table: docs/inventory.md, "Per-kit MIDI deviation".
- Audio is synthesized by the TD-17 module (sample playback), not acoustic.
  Name this limitation; ENST-Drums (~1 h acoustic) is a later transfer sanity check.
- Hi-hat pedal position is MIDI CC4 (from the GMD capture). Whether it is
  continuous or bimodal is decided by docs/inventory.md, not assumed.
- Cymbal chokes are polyphonic aftertouch on Roland kits; whether they survived
  in the MIDI is decided by docs/inventory.md.

## Drum map
drumjepa/drum_map.py: DRUM_MAP_V1, 14 classes:
kick, snare, rimshot, crossstick, tom_hi, tom_mid, tom_lo, hh_closed, hh_open,
hh_pedal, crash1, crash2, ride, ride_bell. Tom rims -> parent tom, hat edge hits ->
parent hat, ride edge -> ride (NOT crash: it's the ride's timbre being crashed).
K9 projection (Magenta convention) exists for comparison with ADT literature.
Rationale: every collapse of two pitches makes the audio less determined by the
action, which corrupts Q1; keep timbrally distinct zones separate.
MAP_VERSION is stamped into every cached tensor and run config. Bump on change.

## Research questions
Q1 — What does the state carry when the action nearly determines the audio?
  Prediction: an action-only predictor f(a_{t+1}) matches the full model on
  prediction loss except on kit-swapped inputs, where the full model wins.
Q2 — Is kit identity recovered as a latent?
  Prediction: linear probe recovers kit from s_t; counterfactual swap (same action,
  s_t from kit B) moves the prediction toward kit B; held-out kits embed sensibly.
Secondary (only if aftertouch survived): 2x2 of {modulates state, injects content}
  x {continuous, discrete}: velocity (continuous, inject) vs choke (discrete,
  modulate). Ziyu's hypothesis: state-modulating actions are easier for the world
  model. Neil's old continuous-vs-discrete hypothesis is dropped (velocity F1 in
  Music-JEPA was the worst metric, which contradicts it).
Rejected: hi-hat pedal as an analogue of piano sustain pedal (different scope,
gating, and usage). Do not reintroduce.

## Experiment blocks and gates (sequential; do not skip)
E1 dynamics sanity: perturbation win rates (state time-shift, random state,
   action time-shift, random action, KIT SWAP). Figure 2 / Table 1 format from the
   paper. Gate: win rates clearly above 0.5. If not, debug; nothing else is valid.
E2 state ablation: full vs action-only predictor on held-out sequences,
   kit-swapped inputs, post-ring-out segments.
E3 kit latent: linear + MLP probes for kit, counterfactual-swap geometry, held-out kits.
E4 inverse recovery: amortized inverse h(s_t, s_{t+1}, a_t) + decoder;
   onset/velocity F1 and hi-hat estimate.
E5 content probes: linear vs MLP recovery of class/velocity/timing from s_t,
   drum-JEPA vs AO-JEPA (audio-only masked JEPA, no actions — must reimplement).
E6 SIGReg swap; characterize the failure mode if any.
Baselines that matter most: (1) action-only predictor, (2) AO-JEPA.
Also: random-init encoder; a small supervised ADT model as transcription ceiling.

## Current state (Sep 2026)
Done: repo scaffold, drum map v1, download/extract/inventory scripts, venv with
torch on MPS, MIDI + CSV downloaded, inventory run (docs/inventory.md exists —
READ IT: check for UNMAPPED pitches, aftertouch presence, CC4 verdict, density).
Done 2026-09-04: inventory verdicts read (no aftertouch -> choke eval is off;
CC4 has 85% mid-range mass; 3 unmapped pitches explained as per-kit remaps,
map stays v1); configs/kits_v1.yaml filled; scripts/kit_midi_diff.py added.
Done 2026-09-04 (evening): audio zip downloaded and sha-verified; 14 kits extracted
to data/audio (14,826 wav, 43 GB, 44.1 kHz mono). Zip kept in data/ (1.3 TB free).
drumjepa/features.py + scripts/build_cache.py + drumjepa/dataset.py +
scripts/bench_loader.py written; full cache build launched (data/build_cache.log).
Warm-up LeJEPA run stopped at epoch 60/100 by a kernel panic; goal (clean curve,
MPS experience) met, not resumed.
Done 2026-09-05: cache v1 built (22 GB). drumjepa/model.py (18.8M student params),
scripts/train.py (--overfit, --resume), tests/ (15 pass). Overfit-32 passed. First
run drumjepa_v1: 20 epochs on 6 kits, no collapse. E1 PASSED (docs/experiments.md,
docs/e1/): kit swap 0.993, random action 0.993, action shift 0.979, random state
0.973, state shift 0.594. Design decisions and caveats: notes/decisions.md.
E2 done 2026-09-05 (docs/experiments.md, docs/e2/): action-only baseline
(drumjepa_v1_actiononly, use_state false) loses to the full model on CLEAN held-out
inputs (err 0.144 vs 0.095, win 0.78, both splits), not only on kit swap. Q1's
"near tie" prediction is refuted: s_t carries information beyond kit identity
(kit swap costs the full model 0.02; the rest of the 0.05 gap survives it). Ring-out
subsets too small (42/36) to conclude.
E3 done 2026-09-06 (docs/experiments.md, docs/e3/): kit is linearly readable from
s_t (0.92) but raw mel does better (0.996) and random-init nearly matches (0.91), so
the probe does not count. The counterfactual geometry does: kit-vector transfer 0.73
vs 0.11 (random) / 0.30 (raw mel); target-fixed predictor swap 0.99 train-B, 0.89
held-out-B. Held-out kits separable and mostly placed as predicted.
E4+E5 done 2026-09-06 (docs/experiments.md, docs/e4/, docs/e5/, docs/e3/aojepa/).
AO-JEPA baseline trained (drumjepa_v1_aojepa, use_action false). Findings: (1) the
kit geometry is built by masked JEPA training, not by actions (AO-JEPA transfer 0.58
vs 0.73); (2) action content is LESS recoverable from drum-JEPA's state than from an
untrained encoder (E4 onset F1 0.12 vs 0.25; E5 linear probe 0.22 vs 0.37), and
AO-JEPA sits between (0.16 / 0.28). Mechanism: f gets a_{t+1}, so the state never
has to carry it. Q1 answer: the state carries what the action does not determine and
drops what it does. Next: discuss with Ziyu whether that is the intended latent;
candidate fixes to test: predict s_{t+1} from s_t only with action as auxiliary
target, or a_t-reconstruction regularizer; then seeds.
Option A tested 2026-09-06 (drumjepa_v1_auxrec, aux_rec 1.0; docs/optionA/): action
content restored and beats both controls (E5 0.63 vs 0.22; E4 0.46 vs 0.12; held-out
0.48 vs raw mel 0.28) BUT prediction degrades (normalized error 0.21 vs 0.08; E2
advantage over action-only gone; E1 kit swap 0.99 -> 0.94). A tradeoff at this
weight.
Sweep done 2026-09-06 (docs/optionA/sweep/): aux_rec 0.1 is the sweet spot: E5 onset
0.58 train / 0.40 held-out (raw mel 0.58 / 0.28, random 0.37 / 0.32) AND normalized
prediction error 0.038 vs 0.081 unregularized, E1 state win rates up. 0.3 and 1.0
keep the content but prediction error jumps 5x.
Seeds done 2026-09-07 (docs/optionA/seeds/): 3 seeds each at aux_rec 0 and 0.1. No
overlap on normalized error (0.077-0.105 vs 0.026-0.038), E5 onset train (0.22-0.27
vs 0.58-0.66) or held-out (0.19-0.21 vs 0.25-0.40), E1 random state, E2 win rate.
The option A headline is seed-robust. Still single-seed: sweep points 0.3/1.0,
AO-JEPA, E3, E4. Next: email Ziyu (draft exists) with E1 figure + seeds table.
Follow-ups done 2026-09-07 (docs/followups.md): E4 on 0.1 passes all seeds; 40-epoch
baseline does not close the gap and sheds more content; held-out swap improves across
seeds but held-out E1 kit swap is identical; fixed-readout eval saturates under the
training mask, use the full-mask ratio (0.92-1.00 vs 0.80-0.88); mel-reconstruction
control (3 seeds) halves normalized error and improves the ratio modestly (0.87-0.91),
action target more (0.92-1.00, separated). Error / teacher variance retired for
cross-model claims. Headline now: "a weak reconstruction term keeps input content in
the state and helps prediction; the action target is the best target tested." Next:
email Ziyu.
Next:
1. (done) Fill configs/kits_v1.yaml, commit, push.
2. Warm-up: warmup/lejepa_inet10.py (LeJEPA minimal, ViT-S/8, Imagenette,
   device-agnostic). Smoke test with --epochs 1 --max-steps 20, then 100 epochs,
   then 800 overnight if it fits. Reference: 90.7% at 800 epochs on one GPU.
   Goal is a clean monotone curve and MPS experience, not the number.
3. (done) Audio extracted. Delete data/e-gmd-v1.0.0.zip if space is tight.
4. (code done, build running) Preprocessing: cache log-mel (200x229), drumroll (200x14), CC4 track (200,),
   with sequence_id, kit_id, onset-density bin, MAP_VERSION, as memory-mapped
   arrays per split. Benchmark the dataloader before writing the model.
5. Port the model in this order, each tested on a synthetic batch:
   state encoder + EMA copy + LN -> action encoder -> state predictor with
   cross-attn -> action predictor -> loss/loop -> overfit test on 32 segments
   (loss must go to ~0) -> real run on 6-kit subset.
6. Log both loss terms AND per-dimension embedding variance every epoch
   (EMA loss going to zero is ambiguous; variance shrinking = collapse).

## Compute
MacBook Pro M5 Max, 48 GB unified memory. Use PyTorch on MPS (not MLX) so the
Music-JEPA code drops in when released. Expect 5-10x slower than an A100; the
19M-param model on ~60 h of audio is ~1-2 days per 20-epoch run here. Full
ablation sweeps go to lab GPUs. Log every MPS incompatibility and fix in
notes/mps_fixes.md (bf16 autocast, unsupported ops, slow attention kernels).

## Conventions
- Python, PyTorch. Keep module names shaped like the paper's Algorithm 1
  (Es_stu, Es_tea, Ea_stu, Ea_tea, f, g) so it diffs cleanly against their
  release later.
- Config in YAML under configs/; every run config records MAP_VERSION and
  split_version.
- data/, third_party/, checkpoints/ are gitignored. Commit docs/inventory.md.
- Prose (README, docs, notes) follows the Google developer style guide.
- Be direct and critical. If a design choice is wrong, say so.

## Reading list (priority order)
Music-JEPA (2607.22000); E-GMD paper (Callender, Hawthorne, Engel 2020,
2004.00188); GrooVAE / Groove MIDI Dataset (Gillick et al. 2019); I-JEPA
(Assran et al. 2023); LeCun 2022 position paper (world-model sections);
Klindt, LeCun & Balestriero 2026 (2605.26379, identifiability — framing only);
Riou et al. 2024 (audio JEPA design choices); LeJEPA (2511.08544, for E6);
DINO-WM; Audio-JEPA (2507.02915, AO-JEPA baseline); Wu et al. 2018 ADT review.
