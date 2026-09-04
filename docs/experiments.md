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
