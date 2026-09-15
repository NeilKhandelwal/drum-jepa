# Related work and theoretical grounding

Checked 2026-09-15 against arXiv/ACL pages. Grouped by which part of the project
each item supports. "Verified" means the paper exists with the stated authors and
claim; summaries are from abstracts and full-text skims, not close readings.

## The base model

- **Music-JEPA** (Wang, Fang, LeCun, arXiv 2607.22000, July 2026). Verified. Piano
  audio as state, pianoroll + sustain as action, ViT state/action encoders, masked
  state predictor with cross-attention to actions, action predictor, EMA teachers
  with stop-grad, LayerNorm, loss_s + 0.5 loss_a. Evaluated on MAESTRO for beat
  tracking, composer ID, key, transcription-by-planning. Reports an AO-JEPA baseline
  with much lower perturbation sensitivity (0.576 vs 0.929). Does NOT probe what the
  state carries versus what the action provides, does not test cross-instrument or
  cross-timbre generalization, and uses no auxiliary or reconstruction loss. Our
  E4/E5, option A and v2 fill exactly those gaps.
- **E-GMD** (Callender, Hawthorne, Engel, arXiv 2004.00188, 2020). Verified. 444 h,
  43 kits, every performance on every kit; built for timbre-robust transcription.
  No paper found that uses the same-performance-across-kits structure for
  representation learning or counterfactual analysis (searched 2026-09-15;
  Break-the-Beat, 2605.14555, does MIDI-to-drum synthesis with reference-timbre
  conditioning and does not use E-GMD's kit structure).

## Q1: what the state carries when the action nearly determines the audio

Theory. With the action given to the predictor, the loss-minimizing state is a
sufficient statistic for s_{t+1} given a_{t+1}; anything the action already
determines is redundant for prediction and can be dropped. Two literatures say this
in different words:

- **Predictive vs control sufficiency.** Kim, "Latent State Design for World Models
  under Sufficiency Constraints" (arXiv 2605.01694, May 2026). Verified. Proposition
  2: predictive sufficiency does not imply control sufficiency; a latent can predict
  future observations while discarding distinctions that matter for action. Our
  state is predictively sufficient (E1, E2) and drops the hits (E4, E5): a concrete
  instance.
- **Endogenous vs exogenous factors.** Thomas et al., "Independently Controllable
  Features" (arXiv 1703.07718, 2017); Efroni et al., "Provable RL with Exogenous
  Distractors via Multistep Inverse Dynamics" (arXiv 2110.08847, ICLR 2022); Lamb et
  al., "Guaranteed Discovery of Control-Endogenous Latent States with Multi-Step
  Inverse Models" (arXiv 2207.08229, TMLR 2023). All verified. That line uses inverse
  dynamics to KEEP the controllable (endogenous) factor and DISCARD exogenous noise.
  Our setting inverts it: the action is an input, so the forward objective keeps the
  exogenous factor (kit) and discards the endogenous one (hits). Same machinery,
  opposite regime; cite as the contrast.
- **Information bottleneck / minimal sufficient statistics** (Tishby et al. 1999;
  Still, "Information Bottleneck Approach to Predictive Inference", Entropy 2014).
  Background for "sufficient statistic" language; no need to lean on it.
- **Prediction makes latents action-relevant.** Yeom et al., "What Makes Video World
  Model Latents Action-Relevant: Prediction over Reconstruction" (arXiv 2606.07687,
  June 2026). Verified. Claims prediction objectives concentrate the latent on
  action-consequential features and discard action-invariant ones. Our result is the
  complementary regime: when the action is provided and nearly determines the
  observation, prediction makes the latent action-INDIFFERENT and keeps the
  invariant factor. Worth citing as the tension.

Grounded restatement of Q1: "Under an action-conditioned predictive objective,
does the state encode the action-determined (endogenous) content of the
observation, the action-independent (exogenous) factor, or both? Sufficiency
arguments predict the exogenous factor only; E-GMD lets us measure each side
separately because the exogenous factor (kit) is labeled and the endogenous one
(hits) is the action."

## Option A: the action-reconstruction auxiliary, and the 2026 inverse-dynamics wave

The technique is not new; the finding around it is. All verified:

- Ivashkov, Balestriero, Schölkopf, "Sensorimotor World Models: Perception for
  Action via Inverse Dynamics" (arXiv 2606.20104, June 2026). Forward + inverse
  dynamics loss on consecutive embeddings; without it the end-to-end encoder
  collapses; with it the latent keeps controllable factors and ignores distractors.
- Zhang et al., "Delta-JEPA" (arXiv 2606.31232, June 2026). Inverse head on the
  latent difference z_{t+1} - z_t to avoid shortcuts; documents that latent-only
  objectives lose action sensitivity.
- Boylan, Hokamp, "No Gaussian Required: Contrastive Inverse Dynamics for JEPA World
  Models" (arXiv 2608.17542, Aug 2026). Replaces LeWorldModel's SIGReg with a
  contrastive inverse head; proves collapse cannot drive the inverse loss below
  chance.
- Zeng, Ren, Song, "PhyLatent" (arXiv 2608.05720, Aug 2026). Three ways a
  non-collapsed JEPA latent still loses dynamics-relevant structure; five
  training-only auxiliaries.
- "Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic
  Planning" (arXiv 2609.03565, Sep 2026). Inverse dynamics + state alignment.
- Older lineage: Pathak et al., ICM (2017); Jaderberg et al., UNREAL (2017).
- LeWorldModel (Maes et al., arXiv 2603.19312, March 2026), DINO-WM, V-JEPA 2-AC:
  the action-conditioned JEPA context.

How ours differs, and what to say in a writeup:
1. Those papers add the inverse head to STOP COLLAPSE in end-to-end training with no
   EMA. Ours has EMA teachers and does not collapse (std monitors, E1), and the state
   still sheds action content, progressively (follow-up 3). "Shedding without
   collapse" is the finding; it is a sufficiency effect, not a collapse effect.
2. Our variance-covariance control (item 6) separates the two mechanisms directly:
   an anti-collapse regularizer of matched magnitude does not restore content and
   hurts prediction. None of the papers above run that control.
3. Their inverse head reads a transition (o_t, o_{t+1}) and recovers the controllable
   factor; ours reads one state and reconstructs that window's own action content.
   Delta-JEPA's shortcut argument (concatenated endpoints let the target carry
   action cues) is a reason to try a difference-based head here too.
4. Our mel-target control shows any reconstruction target helps; the action target
   helps more. The 2026 papers only compare against no auxiliary.

## Q2: kit identity as a latent, and the cross-kit idea

- **Weakly-supervised disentanglement from pairs.** Locatello et al.,
  "Weakly-Supervised Disentanglement Without Compromises" (ICML 2020, arXiv
  2002.02886). Verified. Pairs of observations that share some factors, with only the
  count of changed factors known, are enough for identifiable disentanglement.
  E-GMD is that setting: two kits on one performance share everything except the
  kit factor (one changed factor, known). Also Shu et al., "Weakly Supervised
  Disentanglement with Guarantees" (ICLR 2020), and the 2026 content-style
  identification line (e.g. arXiv 2605.17827).
- **Probing methodology.** Hewitt & Liang, "Designing and Interpreting Probes with
  Control Tasks" (EMNLP 2019); Belinkov, "Probing Classifiers: Promises,
  Shortcomings, and Advances" (Computational Linguistics 48(1), 2022). Both verified.
  Our rule that a probe counts only if the trained encoder beats a random-init
  encoder and the raw input is the practice they recommend; E3's discounted kit
  probe is the case in point.

Grounded restatement of Q2: "Is the exogenous factor identified as a direction in
state space, in the disentanglement sense: does the vector between two kits'
encodings of one performance transfer to other performances, and does it move the
predictor's output the way a kit change would? Pair-based identifiability results
say a training signal built from same-performance pairs should make it so; E3 asks
whether the plain predictive objective already does (partly: transfer 0.73 vs
0.11/0.30 controls) and v2 says more kits do not make it kit-general."

Cross-kit prediction (context from kit A, target from kit B, same performance and
action) is the pair-based weak supervision of Locatello et al. applied inside the
JEPA objective. No prior work found that does this on E-GMD or on any
action-conditioned JEPA (search 2026-09-15).

## Gaps a reviewer would raise

- No cross-family baseline: every comparison is JEPA vs JEPA. A reconstruction-based
  action-conditioned model on the same cache would show whether shedding is
  JEPA-specific. Yeom et al. (above) argue reconstruction latents keep more, which
  predicts the baseline would NOT shed.
- The readout ratio (docs/followups/item1.md) is ours; pair it with E1 win rates,
  which are Music-JEPA's own format, whenever a claim rests on it.
- Synthesized TD-17 audio, 19M params, three seeds on the headline only.
