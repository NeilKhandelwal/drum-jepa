# drum-jepa — project context for Claude Code

## Who and why
Neil Khandelwal (NYU, Math + CS), working in Yann LeCun's lab with Dr. Ziyu Wang
as day-to-day PI. Ziyu is first author of Music-JEPA (arXiv 2607.22000, July 2026).
Plan agreed with him: get hands-on JEPA training experience on a small non-music
JEPA, then build an action-conditioned drum JEPA. The Music-JEPA code is not
released; the architecture is reimplemented from the paper.

Background from Ziyu: SIGReg did not work well with the Music-JEPA architecture,
which relies on EMA + stop-grad + LayerNorm for anti-collapse. E6 (SIGReg swap) is
therefore low priority.

## The project in one paragraph
Action-conditioned JEPA world model of drum sound on E-GMD, following the
Music-JEPA design, plus one addition of our own: an optional weak
action-reconstruction term on the state (`aux_rec`, option A, docs/optionA/). State x_t = 2-second log-mel spectrogram of drum
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
- We subset by kit, all sequences on each: configs/kits_v1.yaml has 6 train +
  8 held-out kits; configs/kits_v2.yaml has 12 train kits with the SAME 8 held
  out (docs/k12.md). Exact kit_name strings from the CSV.
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
  model. An earlier continuous-vs-discrete hypothesis was dropped (velocity F1 in
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

## Current state (2026-09-15)
Blocks E1-E5 are done, plus option A, its sweep, three seeds, the six follow-ups
(including the mel and variance-covariance controls) and the 12-training-kit
variant. Verdicts in docs/experiments.md; per-block write-ups in docs/e1..e5,
docs/optionA, docs/followups, docs/k12. The headline: the state keeps what the
action does not determine and sheds what it does; a weak action-reconstruction
term (aux_rec 0.1) restores content and improves the scale-free prediction ratio,
the gain comes from the target rather than from regularization, and it does not
generalize to unseen kits even with twice the training kits.

Standing rules learned the hard way:
- Actions come from the canonical kit's MIDI, never the kit's own file.
- Never compare losses or normalized errors across models or caches; use win
  rates, probe F1 and the fixed-readout ratio (notes/decisions.md).
- Overnight runs need the 140 W adapter: the 67 W one loses ~47 W under training.
- Run chains under `caffeinate -i nohup`; each chain script has its own log.

Open:
- E6 SIGReg swap (low priority, see above).
- Held-out generalization: the objective, not the data, is the limit (docs/k12.md).
  Candidate next experiment, pending Ziyu: cross-kit prediction (context from kit
  A, target from kit B, same performance and action).
- Second seed on v2 before building on its velocity-probe held-out failure.

## Compute
MacBook Pro M5 Max, 48 GB unified memory. Use PyTorch on MPS (not MLX) so the
Music-JEPA code drops in when released. A 20-epoch run on 6 kits takes
~75 min (bf16 autocast, batch 128); on 12 kits ~2.5 h. Full ablation sweeps go
to lab GPUs. Log every MPS incompatibility and fix in
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
