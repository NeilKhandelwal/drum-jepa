# Follow-ups to the option A result

Written 2026-09-07 after the three-seed comparison. Not started. Ranked by how much
the headline claim depends on each item. The headline: a weak action-reconstruction
term (aux_rec 0.1) restores action content to the state and improves the world
model's prediction at the same time.

## 1. Scale-free prediction metric (eval only)
Problem: "0.1 improves prediction" rests on error / teacher variance. The aux term
also inflates the teacher's spread (std 1.09 to 1.33), so the normalization is doing
real work, and notes/decisions.md already flags it as an assumption.
Do: train one linear readout on the teacher's embeddings of the next window
(target: next-window onsets, or kit), then score every model's predicted embedding
against that readout. Same space for all models, no variance division. Report
next-window onset F1 and kit accuracy from the prediction, for aux_rec 0, 0.1,
action-only, and AO-JEPA, all three seeds where they exist.
Pass: aux_rec 0.1 beats aux_rec 0 on every seed under the fixed readout.
Fail: the gap vanishes, and the prediction half of the headline is withdrawn.

## 2. Control auxiliary target (one run + evals)
Problem: cannot separate "action content in the state helps" from "any weak
regularizer that spreads the embedding helps."
Do: one run at weight 0.1 whose aux target is not the action, for example the
current window's mel (or a fixed random projection of it), same head shape, same
seed 0, same recipe. Run E1, E2, E5 and item 1's metric.
Pass: the control does not halve the normalized error; the mechanism story stands.
Fail: it does, and the claim becomes "a weak auxiliary term helps," with the action
content as a separate, still-true finding.

## 3. Longer baseline (one run, no new evals)
Problem: E2 recorded the baseline still improving at epoch 19. The aux term may only
converge faster.
Do: aux_rec 0, seed 0, 40 epochs. Evaluate at epoch 20 and 40 (E2 normalized error,
E5 onset probe, item 1's metric).
Pass: the epoch-40 baseline is still clearly behind aux_rec 0.1 at epoch 20.
Fail: it catches up, and the claim is "faster training," not "better world model."

## 4. E4 on the 0.1 model (eval only)
Problem: the E5 probe reads the same pooled tokens the aux head trains on, so its
rise is close to tautological. E4 is a different decoder on un-pooled tokens and it
found the original problem. It was rerun at weight 1.0 (0.464), not at 0.1.
Do: eval_e4 on drumjepa_v1_auxrec01, seeds 0, 1, 2. Train and held-out kits.
Pass: E4 onset F1 above both controls (random 0.251, raw mel 0.250).

## 5. Prediction on unseen kits for the 0.1 model (eval only)
Problem: content on held-out kits is measured; prediction on held-out kits is not.
Do: E3 predictor swap with held-out B, and E1 on held-out-kit sequences, for
drumjepa_v1_auxrec01 seed 0, alongside drumjepa_v1 seed 0.
Pass: held-out swap win rate at or above drumjepa_v1's 0.889.

## Notes
- Three seeds with complete separation is convincing at these effect sizes but is
  not significant by a rank test (3 vs 3). Say "no overlap across three seeds," not
  "significant." Five seeds if a reviewer insists.
- Still in the original plan and unrun: E6 SIGReg swap, and the velocity-vs-choke
  secondary. Neither bears on the headline. Scope decisions, not omissions.
- Order: 1 and 2 before the email to Ziyu. 4 and 5 can run in the same afternoon
  as 1. 3 is overnight.
- Cost: 1, 4, 5 need no training. 2 and 3 are one ~75 min run each plus evals.

## Results 2026-09-07
Full write-ups, tables, and artifacts: docs/followups/item1.md through item5.md.
Every eval and all training runs finished, including mel-control seeds 1 and 2
(scripts/run_followups_chain2.sh, 2026-09-08).

| item | result | one line |
|---|---|---|
| 1, fixed readout | passed, narrower than planned | Under the 75% mask every model's predicted grid scores at its ceiling, so that condition measures encoder content, not prediction. Under full mask the ratio does separate: 0.92-1.00 for aux_rec 0.1 against 0.80-0.88 for aux_rec 0, three seeds, no overlap. |
| 2, control target | fails the literal criterion, changes the reading | The mel-target control halves the normalized error (0.021-0.035 vs 0.079-0.108), but both regularized models enter a regime where kit-swapped error is 26-77x the clean error, so that metric means "easier to predict." On the readout ratio the groups order baseline (0.80-0.88) < mel (0.87-0.91) < action (0.92-1.00), with the action target separated from both. Content on train kits orders the same way, all separated; on held-out kits mel (0.25-0.26) beats baseline (0.19-0.21) and the action target (0.25-0.40) overlaps mel at one seed. Three seeds each. |
| 3, longer baseline | passed | 40 epochs: normalized error 0.066, readout ratio 0.867, E2 win 0.776, none near the action target. The normalized-error dip is entirely spread (raw error 0.097 vs 0.095). Action content keeps falling with training (E5 0.223 to 0.197), so shedding is progressive. |
| 4, E4 on 0.1 | passed | Inverse-model onset F1 0.388-0.440 on train kits, 0.270-0.354 held-out, every seed above both controls (0.25). Velocity MAE 14.8-17.0 is the best of any representation. |
| 5, unseen kits | passed on the criterion, narrowly | Held-out predictor swap 0.917-0.955 against 0.886-0.893, three seeds each, no overlap but adjacent CIs touch. E1 on held-out kits: kit swap 0.839 for both weights, identical; only action-side perturbations improve. Kit-vector transfer between train kits reverses sign across seeds. |

What the follow-ups changed in the reading of the headline:
- The prediction half no longer rests on error / teacher variance. That metric
  rewards spread (items 2 and 3) and is retired for cross-model claims. The
  prediction claim now rests on the full-mask readout ratio and the E2 win rate.
- The content half is not action-specific in kind. Any input reconstruction
  restores onset content; the action target restores more of it on training
  kits (seed-robust) and suggestively more on unseen kits (two of three seeds).
- The prediction gain is also shared in kind: the mel control improves the
  readout ratio modestly, the action target more, with no overlap between them.
  The headline phrase "restores action content" should read "asks the state to
  keep more of its input, with the action as the best target tested."
- The shedding mechanism is progressive with training (item 3) and the fix
  survives a decoder that is not the aux head (item 4).
- "Generalizes to unseen kits" is too strong; "improves the held-out swap" is
  what the data supports (item 5).

Open: a control that cannot collapse, such as a variance-covariance regularizer,
to separate "any reconstruction target" from "any regularizer." Not in this plan.
