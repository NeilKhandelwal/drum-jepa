"""Item 1 of docs/followups.md: prediction quality read out in a fixed space.

    python scripts/eval_readout.py --run-dir runs/drumjepa_v1

E2's cross-model comparison divides the state prediction error by the teacher's
per-dimension variance (notes/decisions.md, "Cross-model prediction errors"). The
aux_rec term inflates that variance, so the normalization does real work and the
"0.1 predicts better" claim depends on it. This eval replaces the division with a
readout: a linear probe trained on the model's OWN teacher embeddings of real,
unmasked x_{t+1} clips, then applied to the grid the predictor produces. Nothing is
divided by anything, and every model is scored in units of onset F1 and kit
accuracy rather than in its own embedding units.

Two readouts, both fit on the train split and frozen:
  onset  the 8x16 token grid pooled over frequency to 8 step vectors, then E5's
         linear onset probe (14 independent logistic regressions) against the
         per-250 ms onset targets of the same window.
  kit    E3's 14-way linear kit probe on the mean over the 128 tokens.

Three scored conditions on test-split transitions, all with E1's masks:
  predicted   f's output at the masked positions, teacher tokens at the visible
              ones. This is the metric.
  ceiling     the all-teacher grid of the true x_{t+1}: what the readout can do
              when prediction is perfect. The ratio predicted/ceiling is the
              scale-free number.
  full_mask   every token predicted (K=1), as in E2.

Writes <run-dir>/readout/{results.json,results.md}.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drumjepa.dataset import SegmentPairs  # noqa: E402
from drumjepa.drum_map import CLASSES_V1  # noqa: E402
from eval_e1 import masks_for  # noqa: E402
from eval_e5 import (N_STEPS, THRESHOLDS, counts_by_seq, f1_bootstrap, f1_counts,  # noqa: E402
                     f1_from_counts, fit_onset, load_teacher, step_features, step_targets,
                     window_subset)
from train import git_commit, pick_device  # noqa: E402

CONDS = ("predicted", "ceiling", "full_mask")
K_CLASSES = len(CLASSES_V1)
MOSTLY_MASKED = 0.5  # a step counts as mostly masked when more than half its 16 tokens are


def clip_pool(tokens):
    """Clip embedding, as in E3: mean over the 128 state tokens, (B, N, d) -> (B, d)."""
    return tokens.mean(1)


def predicted_grid(pred, tgt, mask):
    """The grid the metric reads: f's token where the mask is True, the teacher's where not.

    The visible tokens are the teacher's because the predictor was given them: only the
    masked positions are the model's own guess about x_{t+1}.
    """
    return torch.where(mask[..., None], pred, tgt)


class TransitionItems(Dataset):
    """One transition: the tensors the predictor needs plus the identifying columns."""

    def __init__(self, ds, tidx):
        self.ds, self.tidx = ds, tidx

    def __len__(self):
        return len(self.tidx)

    def __getitem__(self, j):
        i = int(self.tidx[j])
        it = self.ds[i]
        return {k: it[k] for k in ("x_t", "x_t1", "a_t1", "cc_t1")} | {
            "kit_id": it["kit_id"], "seq_idx": it["seq_idx"], "tidx": i}


class ClipItems(Dataset):
    """One clip for the readout's training set: the NEXT window and its own action."""

    def __init__(self, ds, tidx):
        self.ds, self.tidx = ds, tidx

    def __len__(self):
        return len(self.tidx)

    def __getitem__(self, j):
        i = int(self.tidx[j])
        it = self.ds[i]
        return {"x": it["x_t1"], "a": it["a_t1"], "cc": it["cc_t1"],
                "kit_id": it["kit_id"], "seq_idx": it["seq_idx"]}


def encode_clips(ds, tidx, model, dev, amp, bs, workers):
    """Teacher features of real unmasked clips: step vectors, clip embedding, targets."""
    dl = DataLoader(ClipItems(ds, tidx), batch_size=bs, num_workers=workers)
    out = {}
    with torch.inference_mode():
        for b in tqdm.tqdm(dl, mininterval=5):
            with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                z = model.encode_state(b["x"].to(dev), teacher=True)
            z = z.float().cpu()
            out.setdefault("step", []).append(
                step_features(z, model.grid).numpy().astype(np.float16))
            out.setdefault("clip", []).append(clip_pool(z).numpy().astype(np.float16))
            out.setdefault("onset", []).append(
                step_targets(b["a"].numpy(), b["cc"].numpy())["onset"])
            for k in ("kit_id", "seq_idx"):
                out.setdefault(k, []).append(b[k].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def run_predictor(ds, tidx, model, dev, amp, args):
    """Per-condition grids of the model's prediction, reduced to readout features.

    The predicted grid takes f's output at the masked positions and the teacher's
    tokens at the visible ones, for each of the K masks of the transition; the
    ceiling grid is the teacher's own tokens of the true x_{t+1} (mask-independent,
    so one row per transition); the full-mask grid is f's output everywhere.
    """
    chunk = max(1, args.bs // args.K)
    dl = DataLoader(TransitionItems(ds, tidx), batch_size=chunk, num_workers=args.workers)
    out, err = {}, {}
    s1 = s2 = n_tok = 0
    with torch.inference_mode():
        for b in tqdm.tqdm(dl, mininterval=5):
            n = b["tidx"].numel()
            mask = torch.cat([masks_for(model, i, args.K, args.seed, False)
                              for i in b["tidx"].tolist()]).to(dev)
            x_t, x_t1, a_t1, cc_t1 = (b[k].to(dev) for k in ("x_t", "x_t1", "a_t1", "cc_t1"))
            rep = [t.repeat_interleave(args.K, 0) for t in (x_t, a_t1, cc_t1, x_t1)]
            with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                pred, tgt, _ = model._state_branch(*rep, mask)
                full = torch.ones(n, model.n_s, dtype=torch.bool, device=dev)
                pred_f, tgt_f, _ = model._state_branch(x_t, a_t1, cc_t1, x_t1, full)
            pred, tgt, pred_f = pred.float(), tgt.float(), pred_f.float()
            se = (pred - tgt).pow(2).mean(-1)
            m = mask.float()
            err.setdefault("masked_mse", []).append(
                ((se * m).sum(1) / m.sum(1)).view(n, args.K).mean(1).cpu().numpy())
            t = tgt_f.float().flatten(0, 1)
            s1, s2, n_tok = s1 + t.sum(0), s2 + t.pow(2).sum(0), n_tok + t.size(0)
            grids = {"predicted": predicted_grid(pred, tgt, mask),
                     "ceiling": tgt.view(n, args.K, model.n_s, -1)[:, 0],
                     "full_mask": pred_f}
            for c, g in grids.items():
                out.setdefault(f"{c}_step", []).append(
                    step_features(g, model.grid).cpu().numpy().astype(np.float16))
                out.setdefault(f"{c}_clip", []).append(
                    clip_pool(g).cpu().numpy().astype(np.float16))
            # Fraction of each time step's 16 frequency tokens that f had to predict.
            out.setdefault("mask_frac", []).append(
                mask.view(-1, *model.grid).float().mean(2).cpu().numpy().astype(np.float16))
            out.setdefault("onset", []).append(
                step_targets(b["a_t1"].numpy(), b["cc_t1"].numpy())["onset"])
            for k in ("kit_id", "seq_idx"):
                out.setdefault(k, []).append(b[k].numpy())
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["masked_mse"] = np.concatenate(err["masked_mse"])
    out["teacher_var"] = float((s2 / n_tok - (s1 / n_tok) ** 2).mean().cpu())
    return out


def cond_rows(feat, cond, K):
    """Step features, clip embeddings, onset targets and sequence ids for one condition.

    The predicted condition has K rows per transition (one per mask); the ceiling and
    full-mask conditions have one, so their targets are not repeated.
    """
    r = K if cond == "predicted" else 1
    step = feat[f"{cond}_step"].astype(np.float32)
    return {"step": step.reshape(-1, step.shape[-1]),
            "clip": feat[f"{cond}_clip"].astype(np.float32),
            "onset": np.repeat(feat["onset"], r, 0).reshape(-1, K_CLASSES),
            "kit_id": np.repeat(feat["kit_id"], r, 0),
            "step_seq": np.repeat(np.repeat(feat["seq_idx"], r, 0), N_STEPS),
            "clip_seq": np.repeat(feat["seq_idx"], r, 0)}


def onset_probs(rows, scaler, predict):
    return predict(scaler.transform(rows["step"]))


def best_threshold(prob, Y):
    """The threshold of THRESHOLDS with the highest macro-F1, picked on validation."""
    scores = [f1_from_counts(f1_counts(Y, prob >= t)).mean() for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def score_onset(prob, Y, seq, thr, boot, seed):
    pred = prob >= thr
    f1 = f1_from_counts(f1_counts(Y, pred))
    out = {"macro_f1": float(f1.mean()), "per_class_f1": f1.tolist(),
           "threshold": thr, "n_rows": int(len(Y))}
    if boot:
        out["macro_f1_ci"] = f1_bootstrap(counts_by_seq(Y, pred, seq), seed=seed)
    return out


def write_outputs(out_dir, res):
    """results.json and results.md; returns the markdown."""
    o, k = res["onset"], res["kit"]
    ratio = res["ratio"]
    rows = ["| condition | n rows | onset macro-F1 | threshold | kit accuracy |",
            "|---|---|---|---|---|"]
    for c in CONDS:
        ci = (f" [{o[c]['macro_f1_ci'][0]:.3f}, {o[c]['macro_f1_ci'][1]:.3f}]"
              if "macro_f1_ci" in o[c] else "")
        rows.append(f"| {c} | {o[c]['n_rows']} | {o[c]['macro_f1']:.3f}{ci} | "
                    f"{o[c]['threshold']:.2f} | {k[c]['accuracy']:.3f} |")
    md = [f"# Readout prediction metric — {res['run_name']} (epoch {res['epoch']}, "
          f"{res['split']} split, {len(res['train_kits'])} train kits)\n",
          f"Readouts fit on {res['n_train_clips']} train-split clips of the model's own "
          f"teacher encoder, applied to {res['n_transitions']} {res['split']} transitions, "
          f"K={res['K']} masks each at {res['mask_ratio']:.0%} masked, seed {res['seed']}. "
          "Nothing is divided by the teacher's variance: both readouts are frozen linear "
          "maps into onset and kit labels.\n",
          "\n".join(rows), "",
          f"Onset F1 of the predicted grid as a fraction of the ceiling: "
          f"**{ratio['onset']:.3f}**; kit accuracy as a fraction of the ceiling: "
          f"{ratio['kit']:.3f}.\n",
          f"Restricted to the {o['predicted_mostly_masked']['n_rows']} time steps where more "
          f"than {MOSTLY_MASKED:.0%} of the 16 frequency tokens were predicted, the predicted "
          f"grid scores {o['predicted_mostly_masked']['macro_f1']:.3f} against a ceiling of "
          f"{o['ceiling_mostly_masked']['macro_f1']:.3f} on the same steps "
          f"(ratio {ratio['onset_mostly_masked']:.3f}).\n",
          "## Protocol", "",
          f"- Onset readout: the 8x16 teacher token grid pooled over frequency to 8 step "
          f"vectors (256-d), then 14 independent logistic regressions against \"was this class "
          f"struck in this 250 ms step\", fit on the {res['n_train_kit_clips']} train-split "
          f"clips that belong to the {len(res['train_kits'])} train kits (E5's protocol). "
          f"The threshold is picked per condition on the validation split "
          f"({res['n_val_transitions']} transitions) and applied to {res['split']}.",
          f"- Kit readout: a 14-way logistic regression on the clip embedding (mean over the "
          f"128 tokens), fit on all {res['n_train_clips']} train-split clips across "
          f"{res['n_kits']} kits (E3's 14-way probe). Scored transitions come from the train "
          f"kits only, as in E1 and E2, so chance is 1/{res['n_kits']} by class count.",
          "- The readout's training clips are the real, unmasked x_{t+1} of each train-split "
          "transition, encoded by the EMA teacher. No prediction is involved in fitting.",
          f"- For reference, the E2-style masked prediction error on the same transitions is "
          f"{res['masked_mse']:.4f}, and the teacher's mean per-dim variance is "
          f"{res['teacher_var']:.4f} (normalized error {res['masked_mse'] / res['teacher_var']:.4f}).",
          "", "### Per-class onset F1, predicted grid against the ceiling", "",
          "| class | predicted | ceiling |", "|---|---|---|"]
    for i, c in enumerate(CLASSES_V1):
        md.append(f"| {c} | {o['predicted']['per_class_f1'][i]:.3f} | "
                  f"{o['ceiling']['per_class_f1'][i]:.3f} |")
    md = "\n".join(md) + "\n"
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    open(os.path.join(out_dir, "results.md"), "w").write(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/drumjepa_v1")
    ap.add_argument("--split", default="test", choices=["validation", "test"],
                    help="split the metric is reported on; validation is always used "
                         "for the threshold")
    ap.add_argument("--max-train", type=int, default=30000, help="readout train clips, 0 = all")
    ap.add_argument("--max-eval", type=int, default=4000,
                    help="transitions per split, 0 = all")
    ap.add_argument("--K", type=int, default=4, help="masks per transition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=128, help="model batch = transitions x K")
    ap.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <run-dir>/readout")
    args = ap.parse_args()

    dev = pick_device(args.device)
    out_dir = args.out or os.path.join(args.run_dir, "readout")
    os.makedirs(out_dir, exist_ok=True)
    model, cfg, ck = load_teacher(args.run_dir, dev)
    dcfg = cfg["config"]["data"]
    amp = bool(cfg["config"]["amp"]) and dev.type != "cpu"

    # Normalization from the run config, never from the cache's stats.json.
    def dataset(split, kits):
        ds = SegmentPairs(dcfg["cache_dir"], split, seg_hop=dcfg["seg_hop"], kits=kits,
                          mel_mean=cfg["mel_mean"], mel_std=cfg["mel_std"])
        for key in ("MAP_VERSION", "split_version"):
            assert ds.meta[key] == dcfg[key], f"{key}: cache {ds.meta[key]} != run {dcfg[key]}"
        return ds

    # The readout trains on every kit (the kit probe is 14-way, as in E3); the onset
    # probe is then fit on the train-kit rows only, as in E5.
    ds_train = dataset("train", None)
    kits = ds_train.meta["kits"]
    train_ids = [kits.index(k) for k in dcfg["train_kits"]]
    tidx = window_subset(ds_train, np.ones(len(ds_train), bool), args.max_train, args.seed)
    print(f"device={dev} amp(bf16)={amp} readout train clips={len(tidx)}", flush=True)
    tr = encode_clips(ds_train, tidx, model, dev, amp, max(args.bs, 256), args.workers)

    is_train_kit = np.isin(tr["kit_id"], train_ids)
    step_rows = tr["step"][is_train_kit].reshape(-1, tr["step"].shape[-1]).astype(np.float32)
    step_scaler = StandardScaler().fit(step_rows)
    clip_scaler = StandardScaler().fit(tr["clip"].astype(np.float32))
    print("fitting the onset readout", flush=True)
    onset_predict = fit_onset(step_scaler.transform(step_rows),
                              tr["onset"][is_train_kit].reshape(-1, K_CLASSES),
                              "linear", args.seed)
    del step_rows
    print("fitting the kit readout", flush=True)
    kit_clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000).fit(
        clip_scaler.transform(tr["clip"].astype(np.float32)), tr["kit_id"])

    # Scoring runs on the train kits only, as E1 and E2 do.
    feat, n_eval = {}, {}
    for split in dict.fromkeys(["validation", args.split]):
        ds = dataset(split, dcfg["train_kits"])
        idx = np.arange(len(ds))
        if args.max_eval:
            idx = np.sort(np.random.default_rng(args.seed).choice(
                len(ds), min(args.max_eval, len(ds)), replace=False))
        print(f"{split}: predicting {len(idx)} transitions", flush=True)
        feat[split] = run_predictor(ds, idx, model, dev, amp, args)
        n_eval[split] = int(len(idx))

    res = {"run_dir": args.run_dir, "run_name": cfg["config"]["run"]["name"],
           "epoch": ck["epoch"], "step": ck["step"], "split": args.split, "seed": args.seed,
           "K": args.K, "mask_ratio": model.mask_ratio, "kits": kits,
           "train_kits": dcfg["train_kits"], "n_kits": len(kits),
           "n_train_clips": int(len(tr["kit_id"])),
           "n_train_kit_clips": int(is_train_kit.sum()),
           "n_transitions": n_eval[args.split],
           "n_val_transitions": n_eval["validation"],
           "max_train": args.max_train, "max_eval": args.max_eval,
           "masked_mse": float(feat[args.split]["masked_mse"].mean()),
           "teacher_var": float(feat[args.split]["teacher_var"]),
           "classes": CLASSES_V1, "git_commit": git_commit(),
           "onset": {}, "kit": {}, "ratio": {}}

    rows = {s: {c: cond_rows(feat[s], c, args.K) for c in CONDS} for s in feat}
    # One threshold per condition, picked on validation and applied to the scored split.
    thr = {c: best_threshold(onset_probs(rows["validation"][c], step_scaler, onset_predict),
                             rows["validation"][c]["onset"]) for c in CONDS}
    prob = {c: onset_probs(rows[args.split][c], step_scaler, onset_predict) for c in CONDS}
    for c in CONDS:
        r = rows[args.split][c]
        res["onset"][c] = score_onset(prob[c], r["onset"], r["step_seq"], thr[c], True, args.seed)
        res["kit"][c] = {"accuracy": float(
            (kit_clf.predict(clip_scaler.transform(r["clip"])) == r["kit_id"]).mean()),
            "n_rows": int(len(r["kit_id"]))}

    # Extra column: the time steps whose 16 frequency tokens were mostly predicted. The
    # ceiling's steps are repeated once per mask so both conditions score the same rows.
    sel = feat[args.split]["mask_frac"].reshape(-1) > MOSTLY_MASKED
    pr = rows[args.split]["predicted"]
    ceil_step = np.repeat(feat[args.split]["ceiling_step"], args.K, 0).astype(np.float32)
    ceil_prob = onset_predict(step_scaler.transform(ceil_step.reshape(-1, ceil_step.shape[-1])))
    res["onset"]["predicted_mostly_masked"] = score_onset(
        prob["predicted"][sel], pr["onset"][sel], pr["step_seq"][sel], thr["predicted"],
        False, args.seed)
    res["onset"]["ceiling_mostly_masked"] = score_onset(
        ceil_prob[sel], pr["onset"][sel], pr["step_seq"][sel], thr["ceiling"], False, args.seed)
    res["ratio"] = {
        "onset": res["onset"]["predicted"]["macro_f1"] / res["onset"]["ceiling"]["macro_f1"],
        "kit": res["kit"]["predicted"]["accuracy"] / res["kit"]["ceiling"]["accuracy"],
        "onset_mostly_masked": (res["onset"]["predicted_mostly_masked"]["macro_f1"] /
                                res["onset"]["ceiling_mostly_masked"]["macro_f1"])}
    print(write_outputs(out_dir, res))


if __name__ == "__main__":
    main()
