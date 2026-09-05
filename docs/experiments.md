# Experiment plan

Gates are sequential. Don't start a block until the previous gate passes.

## Q1 — What does the state carry when the action nearly determines the audio?
Prediction: an action-only predictor f(a_{t+1}) matches the full model f(s_t, a_{t+1})
on prediction loss, except on kit-swapped inputs where the full model wins.

## Q2 — Is kit identity recovered as a latent?
Prediction: linear probe recovers kit from s_t; counterfactual swap (same action,
s_t from kit B) moves the prediction toward kit B; held-out kits embed sensibly.

## Secondary — velocity vs choke (2x2 modulate/inject × continuous/discrete)
Only if aftertouch survived (see docs/inventory.md). Velocity MAE from the
amortized inverse; prediction-loss sensitivity to choke-vs-no-choke on segments
with a ringing open hi-hat.

## Blocks
- E1 dynamics sanity: perturbation win rates (state/action time-shift, random,
  kit swap). Gate: win rates clearly above 0.5.
- E2 state ablation: full vs action-only, on held-out sequences, kit-swapped
  inputs, post-ring-out segments.
- E3 kit latent: linear + MLP probes, counterfactual geometry, held-out kits.
- E4 inverse recovery: onset/velocity F1 and hi-hat estimate via amortized inverse.
- E5 content probes: linear vs MLP, drum-JEPA vs AO-JEPA.
- E6 SIGReg swap; characterize failure mode if any.

## Results

### E1 — passed (2026-09-05, run drumjepa_v1, epoch 19, validation split, 6 train kits)
Win rate = fraction of transitions where the clean prediction error is lower than
the error with one input corrupted; K=4 fixed multi-block masks per transition;
95% CI from a cluster bootstrap over sequences. Full table, per-kit kit-swap
breakdown, figure, training curve and run config: docs/e1/.

| perturbation | win rate [95% CI] |
|---|---|
| state time-shift (adjacent 2 s, same kit) | 0.594 [0.566, 0.620] |
| random state | 0.973 [0.968, 0.977] |
| action time-shift | 0.979 [0.966, 0.989] |
| random action | 0.993 [0.990, 0.995] |
| kit swap | 0.993 [0.987, 0.997] |

Kit swap, the perturbation the project rests on, is above 0.98 on every train
kit. State time-shift is the weakest by design: the adjacent 2 s of the same
recording is a plausible state, and the win rate is above 0.5 but not by much.
Training: 20 epochs, 12,400 steps, ~3.5 min/epoch on M5 Max; final train loss_s
0.094, validation loss_s 0.098; embedding std rose from 0.62 to 1.09 and held (no
collapse). Gate passed; E2 is unblocked.
