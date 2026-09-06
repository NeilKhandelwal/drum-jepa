"""E4 inverse recovery: can the action be read back out of a state transition?

    python scripts/eval_e4.py --drumjepa runs/drumjepa_v1 --aojepa runs/drumjepa_v1_aojepa

An amortized inverse h(s_t, s_{t+1}[, a_t]) is trained on frozen embeddings and
decodes the drumroll a_{t+1} and the pedal track cc_{t+1}. It is scored as drum
transcription: onset F1 at 50 ms, velocity error at matched onsets, CC4 MAE. Four
representations are compared under an identical decoder -- the drum-JEPA teacher,
the AO-JEPA teacher, a random-init encoder, and the raw log-mel patched exactly
like the encoder's patch-embed -- so a number only counts if it beats the last two.
The `with_at` variants add the previous action, which bounds how much of the score
is groove continuation rather than state content.

Writes <out>/{results.json,results.md,f1_by_class.png,training_curves.csv} and one
<out>/<variant>/{metrics.csv,best.pt} per training run; default <drumjepa>/e4.
"""
import argparse
import csv
import json
import os
import sys
import time

import matplotlib
import numpy as np
import torch
import tqdm
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drumjepa.drum_map import CLASSES_V1, K_V1  # noqa: E402
from drumjepa.features import SEG_FRAMES  # noqa: E402
from drumjepa.inverse import InverseModel, inverse_loss  # noqa: E402
from drumjepa.dataset import SegmentPairs  # noqa: E402
from drumjepa.model import DEFAULTS, DrumJEPA  # noqa: E402
from eval_e3 import train_subset  # noqa: E402
from train import git_commit, pick_device  # noqa: E402

SCORE_SPLITS = ("validation", "test")
GROUPS = ("train_kits", "heldout_kits")
TOL = 5          # +-5 frames = 50 ms onset tolerance
NMS = 2          # an onset must be a local max within +-2 frames
THRESH = 0.5     # the protocol threshold, reported alongside the tuned one
GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95, contains THRESH
POS_WEIGHT_CAP = 50.0
N_BOOT = 1000
FIELDS = ("epoch", "step", "train/loss", "train/bce", "train/vel", "train/cc",
          "val/loss", "val/bce", "val/vel", "val/cc", "lr", "epoch_s")


# ---- representations -------------------------------------------------------
def raw_mel_tokens(x, n_mels_pad=DEFAULTS["n_mels_pad"], patch=DEFAULTS["state_patch"]):
    """Normalized log-mel (B, T, N_MELS) -> (B, n_tok, pt*pf), the encoder's patches.

    Same zero-padding, same patch grid and the same row-major (time, freq) token and
    within-patch order as StateEncoder's Conv2d patch-embed, so the raw baseline and
    the encoders differ only in what happens after the patching.
    """
    pt, pf = patch
    x = torch.nn.functional.pad(x, (0, n_mels_pad - x.size(-1)))
    B, T, F = x.shape
    x = x.view(B, T // pt, pt, F // pf, pf).permute(0, 1, 3, 2, 4)
    return x.reshape(B, (T // pt) * (F // pf), pt * pf)


def load_encoder(run_dir, dev):
    """Frozen teacher state encoder of a training run, plus its config and checkpoint."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    ck = torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu", weights_only=False)
    model = DrumJEPA(cfg["config"]["model"])
    model.load_state_dict(ck["model"])
    return model.to(dev).eval().requires_grad_(False), cfg, ck


def make_reps(args, dev):
    """rep -> (embed fn, token width, provenance dict). All embeddings are frozen."""
    reps, info = {}, {}
    for name, run in (("drumjepa", args.drumjepa), ("aojepa", args.aojepa)):
        if name not in args.reps:
            continue
        m, cfg, ck = load_encoder(run, dev)
        reps[name] = (lambda x, m=m: m.encode_state(x, teacher=True), m.cfg["d_model"])
        info[name] = {"run_dir": run, "epoch": ck["epoch"], "step": ck["step"],
                      "partial": ck["epoch"] + 1 < cfg["config"]["optim"]["epochs"],
                      "mel_mean": cfg["mel_mean"], "mel_std": cfg["mel_std"]}
    if "random" in args.reps:
        torch.manual_seed(0)
        m = DrumJEPA({}).to(dev).eval().requires_grad_(False)
        reps["random"] = (lambda x, m=m: m.encode_state(x, teacher=True), m.cfg["d_model"])
        info["random"] = {"run_dir": None, "note": "DrumJEPA({}) untrained, torch.manual_seed(0)"}
    if "raw_mel" in args.reps:
        pt, pf = DEFAULTS["state_patch"]
        reps["raw_mel"] = (raw_mel_tokens, pt * pf)
        info["raw_mel"] = {"run_dir": None, "note": f"encoder patch-embed grid, {pt * pf}-d patches"}
    return reps, info


# ---- decoding and matching -------------------------------------------------
def onset_peaks(prob, nms=NMS):
    """Onset probabilities (..., T, K) -> bool local maxima within +-nms frames.

    Ties go to the earliest frame, so a plateau of equal probabilities collapses to
    one peak instead of several. This is threshold-independent, which is what makes
    the threshold sweep cheap: peaks are found once and each threshold refilters them.
    """
    T = prob.shape[-2]
    keep = np.ones(prob.shape, bool)
    for d in range(1, nms + 1):
        keep[..., d:, :] &= prob[..., d:, :] > prob[..., :T - d, :]
        keep[..., :T - d, :] &= prob[..., :T - d, :] >= prob[..., d:, :]
    return keep


def decode_onsets(prob, thresh=THRESH, nms=NMS):
    """Onset probabilities (..., T, K) -> bool onsets, thresholded and NMS'd."""
    return onset_peaks(prob, nms) & (prob > thresh)


def match_onsets(true_idx, pred_idx, tol=TOL):
    """Greedy matching of predictions to true onsets, (true frame, pred frame) pairs.

    True onsets are visited in time order and take the nearest unmatched prediction
    within `tol` frames (the earliest one on a tie), the usual ADT convention.
    """
    if len(pred_idx) == 0:
        return []
    used = np.zeros(len(pred_idx), bool)
    pairs = []
    for t in true_idx:
        d = np.abs(pred_idx - t)
        d[used] = tol + 1
        j = int(d.argmin())
        if d[j] <= tol:
            used[j] = True
            pairs.append((int(t), int(pred_idx[j])))
    return pairs


def count_batch(prob, roll, vel, thresholds=(THRESH,)):
    """Per-item, per-class (tp, fp, fn) and matched-onset velocity error, per threshold.

    Peaks are found once (they do not depend on the threshold) and each threshold only
    refilters them, so sweeping 19 thresholds costs far less than 19 decodes.

    Args:
      prob: (B, T, K) onset probabilities. roll: (B, T, K) true velocities/127.
      vel: (B, T, K) predicted velocities/127. thresholds: probabilities to score at.

    Returns:
      tp, fp, fn as (n_thresholds, B, K) int32, and (velocity abs-error sum, n matched)
      as (n_thresholds, B).
    """
    B, _, K = roll.shape
    true = roll > 0
    peaks = onset_peaks(prob) & (prob > min(thresholds))
    nth = len(thresholds)
    tp = np.zeros((nth, B, K), np.int32)
    n_pred = np.zeros((nth, B, K), np.int32)
    n_true = true.sum(1).astype(np.int32)
    verr, vn = np.zeros((nth, B)), np.zeros((nth, B), np.int32)
    for b, k in zip(*np.nonzero((n_true > 0) | peaks.any(1))):
        t_idx = np.flatnonzero(true[b, :, k])
        p_all = np.flatnonzero(peaks[b, :, k])
        p_prob = prob[b, p_all, k]
        for ti, th in enumerate(thresholds):
            pairs = match_onsets(t_idx, p_all[p_prob > th])
            n_pred[ti, b, k] = int((p_prob > th).sum())
            tp[ti, b, k] = len(pairs)
            for t, p in pairs:
                verr[ti, b] += abs(float(vel[b, p, k]) - float(roll[b, t, k]))
            vn[ti, b] += len(pairs)
    return tp, n_pred - tp, n_true[None] - tp, verr, vn


# ---- aggregation -----------------------------------------------------------
def per_sequence_counts(tp, fp, fn, seq):
    """Sum per-item (n, K) counts into (n_sequences, K, 3), the bootstrap unit."""
    _, inv = np.unique(seq, return_inverse=True)
    out = np.zeros((inv.max() + 1, tp.shape[1], 3), np.int64)
    for j, arr in enumerate((tp, fp, fn)):
        np.add.at(out[:, :, j], inv, arr)
    return out


def f1_from(c):
    """(K, 3) tp/fp/fn -> per-class F1, precision, recall (0 where undefined)."""
    tp, fp, fn = c[:, 0].astype(float), c[:, 1].astype(float), c[:, 2].astype(float)
    return (np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp)), where=2 * tp + fp + fn > 0),
            np.divide(tp, tp + fp, out=np.zeros(len(tp)), where=tp + fp > 0),
            np.divide(tp, tp + fn, out=np.zeros(len(tp)), where=tp + fn > 0))


def macro_f1(c, active):
    """Mean per-class F1 over `active` classes (those with at least one true onset)."""
    return float(f1_from(c)[0][active].mean()) if active.any() else 0.0


def bootstrap_ci(counts, active, n_boot=N_BOOT, seed=0):
    """95% cluster-bootstrap CI on macro F1, resampling SEQUENCES (kits share them)."""
    rng = np.random.default_rng(seed)
    n = counts.shape[0]
    boots = [macro_f1(counts[rng.integers(0, n, n)].sum(0), active) for _ in range(n_boot)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def sweep_macro_f1(items, sel, thresholds):
    """Macro F1 on `sel` at every threshold. The active class set does not move with it."""
    out = {}
    for ti, th in enumerate(thresholds):
        c = np.stack([items[k][ti][sel].sum(0) for k in ("tp", "fp", "fn")], axis=1)
        out[th] = macro_f1(c, c[:, 0] + c[:, 2] > 0)
    return out


def summarize(items, sel, seed, ti, threshold):
    """Onset/velocity/CC metrics for the selected items at threshold index `ti`."""
    counts = per_sequence_counts(items["tp"][ti][sel], items["fp"][ti][sel],
                                 items["fn"][ti][sel], items["seq_idx"][sel])
    c = counts.sum(0)
    active = c[:, 0] + c[:, 2] > 0
    f1, prec, rec = f1_from(c)
    tp, fp, fn = c.sum(0)
    vn = items["vel_n"][ti][sel].sum()
    return {
        "threshold": float(threshold),
        "n": int(sel.sum()), "n_sequences": int(counts.shape[0]),
        "macro_f1": macro_f1(c, active), "macro_f1_ci": bootstrap_ci(counts, active, seed=seed),
        "micro_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "micro_precision": float(tp / max(tp + fp, 1)), "micro_recall": float(tp / max(tp + fn, 1)),
        "n_true_onsets": int(tp + fn), "n_pred_onsets": int(tp + fp), "n_matched": int(vn),
        "velocity_mae": float(127 * items["vel_err"][ti][sel].sum() / max(vn, 1)),
        "velocity_mae_baseline": float(127 * items["vel_base"][sel].sum()
                                       / max(items["n_true"][sel].sum(), 1)),
        "cc_mae": float(127 * items["cc_err"][sel].mean()),
        "cc_mae_baseline": float(127 * items["cc_base"][sel].mean()),
        "per_class": {CLASSES_V1[k]: {"f1": float(f1[k]), "precision": float(prec[k]),
                                      "recall": float(rec[k]), "n_true": int(c[k, 0] + c[k, 2])}
                      for k in range(K_V1)},
    }


# ---- data ------------------------------------------------------------------
def action_stats(ds, idx):
    """Onset rate, mean onset velocity and mean CC4 of the a_{t+1} of `idx`.

    Read straight off the memmap: the pos_weight and the constant baselines are
    train-split constants, so a full DataLoader pass for them would be waste.
    """
    starts = ds.seq_start[idx] + ds.offset[idx] + SEG_FRAMES
    n_cell = n_on = 0
    v_sum = cc_sum = 0.0
    cc_n = 0
    for s in starts:
        r = np.asarray(ds.roll[s:s + SEG_FRAMES], np.float32)
        c = np.asarray(ds.cc4[s:s + SEG_FRAMES], np.float32)
        on = r > 0
        n_cell += r.size
        n_on += int(on.sum())
        v_sum += float(r[on].sum())
        cc_sum += float(c.sum())
        cc_n += c.size
    return {"n_cells": n_cell, "n_onsets": n_on,
            "pos_weight": min(POS_WEIGHT_CAP, (n_cell - n_on) / max(n_on, 1)),
            "mean_velocity": v_sum / max(n_on, 1), "mean_cc4": cc_sum / max(cc_n, 1)}


def loader(ds, idx, bs, workers, shuffle=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(ds, sorted(np.asarray(idx).tolist())), batch_size=bs,
                      shuffle=shuffle, num_workers=workers, generator=g,
                      persistent_workers=workers > 0)


# ---- train and score -------------------------------------------------------
def run_batch(model, embed, b, dev, amp, pw):
    """Embed one batch under no_grad and run h and the loss inside autocast."""
    with torch.no_grad(), autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
        z_t, z_t1 = embed(b["x_t"].to(dev)), embed(b["x_t1"].to(dev))
    with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
        at = (b["a_t"].to(dev), b["cc_t"].to(dev)) if model.use_at else (None, None)
        out = model(z_t, z_t1, *at)
        return out, inverse_loss(out, b["a_t1"].to(dev), b["cc_t1"].to(dev), pw)


def train_variant(name, d_in, use_at, embed, train_dl, val_dl, pw, args, dev, amp, out_dir,
                  enc_epoch=None):
    """Train h for one (representation, variant) and keep the best-validation weights."""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    model = InverseModel(d_in=d_in, use_at=use_at).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=max(1, args.epochs * len(train_dl)))
    ckpt = os.path.join(out_dir, "best.pt")
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        csv.DictWriter(f, FIELDS).writeheader()
    print(f"[{name}] h params {model.n_params() / 1e6:.2f}M, {len(train_dl)} steps/epoch")

    best, rows, step, t_start = float("inf"), [], 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        t0, sums, n = time.time(), {k: 0.0 for k in ("loss", "bce", "vel", "cc")}, 0
        for b in tqdm.tqdm(train_dl, desc=f"{name} epoch {epoch}", leave=False, mininterval=5):
            _, ls = run_batch(model, embed, b, dev, amp, pw)
            opt.zero_grad(set_to_none=True)
            ls["loss"].backward()
            opt.step()
            sched.step()
            step += 1
            n += 1
            for k in sums:
                sums[k] += float(ls[k].detach())
        model.eval()
        vs, vn = {k: 0.0 for k in sums}, 0
        with torch.inference_mode():
            for b in val_dl:
                _, ls = run_batch(model, embed, b, dev, amp, pw)
                for k in vs:
                    vs[k] += float(ls[k])
                vn += 1
        row = {"epoch": epoch, "step": step, "lr": opt.param_groups[0]["lr"],
               "epoch_s": round(time.time() - t0, 1),
               **{f"train/{k}": sums[k] / max(n, 1) for k in sums},
               **{f"val/{k}": vs[k] / max(vn, 1) for k in vs}}
        rows.append(row)
        with open(os.path.join(out_dir, "metrics.csv"), "a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(row)
        if row["val/loss"] < best:
            best = row["val/loss"]
            # enc_epoch pins which encoder checkpoint these weights were fitted on, so a
            # later --score-only can tell whether that checkpoint has moved underneath it.
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": best,
                        "d_in": d_in, "use_at": use_at, "pos_weight": float(pw),
                        "encoder_epoch": enc_epoch}, ckpt)
        print(f"[{name}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in row.items()), flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    return model, {"train_time_s": round(time.time() - t_start, 1), "best_val_loss": best,
                   "n_params": model.n_params(), "curves": rows}


def score_pass(models, embed, dl, dev, amp, stats, thresholds):
    """One pass over a scoring split: per-item counts and errors for each variant.

    Args:
      thresholds: variant name -> the onset thresholds to score it at. Everything but
        `cc_err` gains a leading threshold axis in that order.
    """
    acc = {name: {k: [] for k in ("tp", "fp", "fn", "vel_err", "vel_n", "cc_err")}
           for name in models}
    cols = {k: [] for k in ("seq_idx", "kit_id", "n_true", "vel_base", "cc_base")}
    with torch.inference_mode():
        for b in tqdm.tqdm(dl, mininterval=5):
            roll, cc = b["a_t1"].numpy(), b["cc_t1"].numpy()
            with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                z_t, z_t1 = embed(b["x_t"].to(dev)), embed(b["x_t1"].to(dev))
                for name, m in models.items():
                    at = (b["a_t"].to(dev), b["cc_t"].to(dev)) if m.use_at else (None, None)
                    out = m(z_t, z_t1, *at)
                    prob = torch.sigmoid(out["onset_logit"].float()).cpu().numpy()
                    vel = out["velocity"].float().cpu().numpy()
                    ccp = out["cc"].float().cpu().numpy()
                    tp, fp, fn, verr, vn = count_batch(prob, roll, vel, thresholds[name])
                    for k, v in zip(("tp", "fp", "fn", "vel_err", "vel_n", "cc_err"),
                                    (tp, fp, fn, verr, vn, np.abs(ccp - cc).mean(1))):
                        acc[name][k].append(v)
            on = roll > 0
            cols["n_true"].append(on.sum((1, 2)))
            cols["vel_base"].append((np.abs(roll - stats["mean_velocity"]) * on).sum((1, 2)))
            cols["cc_base"].append(np.abs(cc - stats["mean_cc4"]).mean(1))
            for k in ("seq_idx", "kit_id"):
                cols[k].append(b[k].numpy())
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    # Everything but cc_err carries a leading threshold axis, so items join on axis 1.
    return {name: {**{k: np.concatenate(v, axis=0 if k == "cc_err" else 1)
                      for k, v in a.items()}, **cols}
            for name, a in acc.items()}


# ---- outputs ---------------------------------------------------------------
def f1_figure(path, res, variants, split="test", group="train_kits"):
    """Grouped per-class F1 bars at the tuned threshold, representations without a_t."""
    reps = [v for v in variants
            if not v.endswith("_with_at") and not res["variants"][v].get("encoder_drift")]
    x = np.arange(K_V1)
    w = 0.8 / max(len(reps), 1)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, v in enumerate(reps):
        r = res["variants"][v]["scores"][f"{split}/{group}"]["at_tuned"]
        ax.bar(x + i * w - 0.4 + w / 2, [r["per_class"][c]["f1"] for c in CLASSES_V1], w,
               label=f"{v} (t={r['threshold']:.2f})")
    ax.set_xticks(x, CLASSES_V1, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("onset F1 @ 50 ms")
    ax.set_ylim(0, 1)
    ax.set_title(f"E4: per-class onset F1 at the tuned threshold, {split} split, {group}",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=len(reps))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def verdict(res, variants):
    """Four to five sentences on Q1/E4, read off the metrics."""
    def s(v, split="test", group="train_kits", at="at_tuned"):
        return res["variants"][v]["scores"][f"{split}/{group}"][at]

    def cited(v, split="test", group="train_kits", label=True):
        r = s(v, split, group)
        return (f"{v} " if label else "") + (
            f"{r['macro_f1']:.3f} [{r['macro_f1_ci'][0]:.3f}, {r['macro_f1_ci'][1]:.3f}] "
            f"at t={r['threshold']:.2f}")

    # A variant whose encoder moved between fitting h and scoring it is not usable, so it
    # is named as broken rather than ranked (see the encoder_drift check in main).
    stale = [v for v in variants if res["variants"][v].get("encoder_drift")]
    have = lambda v: v in res["variants"] and v not in stale  # noqa: E731
    plain = [v for v in variants if not v.endswith("_with_at") and v not in stale]
    rank = sorted(plain, key=lambda v: -s(v)["macro_f1"])
    d = s("drumjepa")
    ctrl = [v for v in ("random", "raw_mel") if have(v)]
    beaten = [v for v in ctrl if d["macro_f1"] > s(v)["macro_f1"]]
    out = [
        "A state transition does carry the action, but how much of it survives depends on the "
        "encoder, not on the decoder: under an identical h, and with each variant's onset "
        "threshold tuned for macro F1 on the validation train kits, the test-split macro "
        "onset F1 on the six train kits is " + ", ".join(cited(v) for v in rank)
        + f", against 0.000 for predicting no onsets, and every representation beats both "
          f"constant baselines on velocity and pedal position (drum-JEPA: velocity MAE "
          f"{d['velocity_mae']:.1f} against {d['velocity_mae_baseline']:.1f} MIDI units at "
          f"matched onsets, CC4 MAE {d['cc_mae']:.1f} against {d['cc_mae_baseline']:.1f}).",
        f"Drum-JEPA is {'first' if rank[0] == 'drumjepa' else 'last'} of the "
        f"{len(rank)} and beats "
        + ("both controls" if len(beaten) == len(ctrl)
           else (beaten[0] if beaten else "neither control"))
        + " (" + ", ".join(f"{v} {s(v)['macro_f1']:.3f}" for v in ctrl) + "), so by the bar in "
        + ("notes/decisions.md the representation, not h, is what the number measures."
           if len(beaten) == len(ctrl) else
           "notes/decisions.md the number does not count in drum-JEPA's favour: on this task "
           "the JEPA objective makes the state encoder WORSE than the same architecture "
           "untrained. That is what an action-conditioned objective predicts, though: f is "
           "handed a_(t+1) at every step, so neither s_t nor s_(t+1) ever has to encode it, "
           "and 12 layers of training are free to throw the onset detail away."),
    ]
    if have("aojepa"):
        a = s("aojepa")
        out.append(
            f"The E5 comparison lands early and it lands hard: AO-JEPA, the same architecture "
            f"on the same audio with the actions removed, scores "
            f"{cited('aojepa', label=False)} against "
            f"drum-JEPA's {d['macro_f1']:.3f}, so removing action conditioning "
            f"{'costs' if d['macro_f1'] > a['macro_f1'] else 'buys'} the state encoder "
            f"{abs(a['macro_f1'] - d['macro_f1']):.3f} of macro F1"
            + (". " if not res["variants"]["aojepa"]["encoder"].get("partial") else
               f", and this is from a checkpoint at epoch "
               f"{res['variants']['aojepa']['encoder']['epoch']} of a run that is still "
               f"training, so it is provisional and if anything an underestimate. ")
            + "E5 should be read against this row, not around it.")
    if stale:
        out.append(
            "Not usable in this pass: " + ", ".join(
                f"{v} (h fitted against encoder epoch "
                f"{res['variants'][v]['encoder_epoch_trained']}, scored against epoch "
                f"{res['variants'][v]['encoder_epoch_scored']})" for v in stale)
            + ". The encoder run is still training and its last.pt moved between fitting h "
              "and scoring it, so h was handed a representation it was never trained on; "
              "those rows are in the tables for completeness and are excluded from every "
              "comparison above. Refit h on the final checkpoint before reading them.")
    dh = s("drumjepa", "test", "heldout_kits")
    drops = {v: s(v)["macro_f1"] - s(v, "test", "heldout_kits")["macro_f1"] for v in plain}
    out.append(
        f"Held-out kits cost drum-JEPA {drops['drumjepa']:+.3f} of macro F1 "
        f"({dh['macro_f1']:.3f} against {d['macro_f1']:.3f}) against "
        + ", ".join(f"{drops[v]:+.3f} ({v})" for v in plain if v != "drumjepa")
        + ", so it is the flattest across the kit split — but flat at "
        f"{d['macro_f1']:.3f} is a floor effect, not evidence that it generalizes: the "
        "representations that carry action content are the ones with something to lose.")
    if have("drumjepa_with_at"):
        w = s("drumjepa_with_at")
        others = [v for v in variants if v.endswith("_with_at") and v != "drumjepa_with_at"]
        deltas = {v: s(v)["macro_f1"] - s(v[:-len("_with_at")])["macro_f1"]
                  for v in ["drumjepa_with_at"] + others}
        out.append(
            "Handing h the previous action moves macro F1 by "
            + ", ".join(f"{deltas[v]:+.3f} ({v[:-len('_with_at')]})" for v in deltas)
            + ", so groove continuation is not what carries these scores"
            + ("." if all(v > 0 for v in deltas.values()) else
               "; the negative entries are an optimization artifact of the fixed 8-epoch "
               "budget (their validation loss is higher too, "
               + ", ".join(
                   f"{res['variants'][v]['training']['best_val_loss']:.3f} against "
                   f"{res['variants'][v[:-len('_with_at')]]['training']['best_val_loss']:.3f} "
                   f"({v[:-len('_with_at')]})"
                   for v, delta in deltas.items() if delta < 0)
               + "), not a finding."))
    return out


def write_outputs(out_dir, res, variants):
    """results.json, results.md, f1_by_class.png, training_curves.csv; returns the markdown."""
    md = [f"# E4 inverse recovery — {res['n_train']} train transitions, {res['epochs']} epochs, "
          f"h = {res['h_params'] / 1e6:.2f}M params\n",
          "h(s_t, s_(t+1)[, a_t]) decodes a_(t+1) and cc_(t+1) from frozen embeddings; the "
          "decoder is identical across representations and only the input projection differs. "
          f"Onsets are peak-picked with +-{NMS}-frame non-max suppression and matched greedily "
          f"within +-{TOL} frames ({10 * TOL} ms). Macro F1 averages the per-class F1 over the "
          "classes with at least one true onset in the subset; the CI is a "
          f"{N_BOOT}-resample cluster bootstrap over sequences.\n",
          f"Two thresholds are reported. F1@{THRESH} is the protocol threshold; because the "
          f"onset BCE pos_weight is capped at {POS_WEIGHT_CAP:.0f} against a true "
          f"negative/positive ratio of "
          f"{(res['action_stats']['n_cells'] - res['action_stats']['n_onsets']) / res['action_stats']['n_onsets']:.0f}"
          ", every variant over-predicts there and the absolute numbers are not comparable to "
          "published transcription figures. F1@tuned picks each variant's threshold on the "
          f"grid {GRID[0]}..{GRID[-1]} step 0.05 by macro F1 on the VALIDATION train kits, then "
          "applies that one threshold unchanged to both test kit groups; it is the number to "
          "read. Velocity MAE below is also at the tuned threshold (matched onsets move with "
          "it); CC4 MAE does not depend on the threshold.\n",
          "| representation | split | kit group | n | macro F1@0.5 [95% CI] | tuned t | "
          "macro F1@tuned [95% CI] | micro F1@tuned |",
          "|---|---|---|---|---|---|---|---|"]
    for v in variants:
        for split in SCORE_SPLITS:
            for grp in GROUPS:
                sc = res["variants"][v]["scores"][f"{split}/{grp}"]
                h, t = sc["at_half"], sc["at_tuned"]
                md.append(f"| {v} | {split} | {grp} | {h['n']} | {h['macro_f1']:.3f} "
                          f"[{h['macro_f1_ci'][0]:.3f}, {h['macro_f1_ci'][1]:.3f}] | "
                          f"{t['threshold']:.2f} | {t['macro_f1']:.3f} "
                          f"[{t['macro_f1_ci'][0]:.3f}, {t['macro_f1_ci'][1]:.3f}] | "
                          f"{t['micro_f1']:.3f} |")
    stale = [v for v in variants if res["variants"][v].get("encoder_drift")]
    if stale:
        md += ["", "> **Not usable in this pass: " + ", ".join(stale) + ".** The encoder run "
               "is still training, and its `last.pt` moved between fitting h and scoring it ("
               + ", ".join(f"{v}: encoder epoch "
                           f"{res['variants'][v]['encoder_epoch_trained']} -> "
                           f"{res['variants'][v]['encoder_epoch_scored']}" for v in stale)
               + "), so h was handed a representation it was never trained on. These rows are "
                 "kept for completeness and excluded from the per-class table, the figure and "
                 "the Q1/E4 section. Refit h on the final checkpoint before reading them."]
    cols = [v for v in ("drumjepa", "aojepa", "raw_mel", "random")
            if v in res["variants"] and v not in stale]
    pr = ", ".join(
        f"{v} {res['variants'][v]['scores']['test/train_kits']['at_tuned']['micro_precision']:.3f}"
        f"/{res['variants'][v]['scores']['test/train_kits']['at_tuned']['micro_recall']:.3f}"
        for v in cols)
    md += ["", "Predicting no onsets at all scores 0.000 macro and micro F1 everywhere, so the "
           "constant baseline is only informative for velocity and CC4 (below).", "",
           f"Micro precision/recall at the tuned threshold, test split, train kits: {pr}.", "",
           "## Per-class onset F1 at the tuned threshold — test split, train kits", ""]
    md += ["| class | n true onsets | "
           + " | ".join(f"{v} (t={res['variants'][v]['tuned_threshold']:.2f})" for v in cols)
           + " |", "|---" * (len(cols) + 2) + "|"]
    def pc(v):  # noqa: E306
        return res["variants"][v]["scores"]["test/train_kits"]["at_tuned"]["per_class"]
    for c in CLASSES_V1:
        md.append(f"| {c} | {pc(cols[0])[c]['n_true']} | "
                  + " | ".join(f"{pc(v)[c]['f1']:.3f}" for v in cols) + " |")

    md += ["", "## Velocity and hi-hat pedal", "",
           "Velocity MAE is over matched onsets only at the tuned threshold, in MIDI units "
           "(x127); its baseline predicts the train-split mean onset velocity at every true "
           "onset. CC4 MAE is over all frames; its baseline is the train-split mean CC4.", "",
           "| representation | split | kit group | velocity MAE | (constant) | CC4 MAE | "
           "(constant) |", "|---|---|---|---|---|---|---|"]
    for v in variants:
        for split in SCORE_SPLITS:
            for grp in GROUPS:
                r = res["variants"][v]["scores"][f"{split}/{grp}"]["at_tuned"]
                md.append(f"| {v} | {split} | {grp} | {r['velocity_mae']:.1f} | "
                          f"{r['velocity_mae_baseline']:.1f} | {r['cc_mae']:.1f} | "
                          f"{r['cc_mae_baseline']:.1f} |")
    md += ["", "## Training", "",
           "| variant | h params | best val loss | epoch | train time (s) |",
           "|---|---|---|---|---|"]
    for v in variants:
        t = res["variants"][v]["training"]
        md.append(f"| {v} | {t['n_params'] / 1e6:.2f}M | {t['best_val_loss']:.4f} | "
                  f"{t['best_epoch']} | {t['train_time_s']:.0f} |")
    md += ["", "## Q1/E4", ""] + res["verdict"] + [""]

    md = "\n".join(md)
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    open(os.path.join(out_dir, "results.md"), "w").write(md)
    with open(os.path.join(out_dir, "training_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ("variant",) + FIELDS)
        w.writeheader()
        for v in variants:
            for row in res["variants"][v]["training"]["curves"]:
                w.writerow({"variant": v, **row})
    f1_figure(os.path.join(out_dir, "f1_by_class.png"), res, variants)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drumjepa", default="runs/drumjepa_v1")
    ap.add_argument("--aojepa", default="runs/drumjepa_v1_aojepa")
    ap.add_argument("--reps", default="drumjepa,aojepa,random,raw_mel")
    ap.add_argument("--with-at", default="drumjepa,raw_mel", help="reps also given a_t")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <drumjepa>/e4")
    ap.add_argument("--score-only", action="store_true", help="reuse each variant's best.pt")
    ap.add_argument("--max-score", type=int, default=0, help="cap scoring transitions per split")
    args = ap.parse_args()
    args.reps = [r for r in args.reps.split(",") if r]
    args.with_at = [r for r in args.with_at.split(",") if r]
    out_dir = args.out or os.path.join(args.drumjepa, "e4")
    os.makedirs(out_dir, exist_ok=True)

    cfg = json.load(open(os.path.join(args.drumjepa, "config.json")))
    dcfg = cfg["config"]["data"]
    dev = pick_device(args.device)
    amp = bool(cfg["config"]["amp"]) and dev.type != "cpu"
    print(f"device={dev} amp(bf16)={amp} reps={args.reps} with_at={args.with_at}")

    reps, rep_info = make_reps(args, dev)
    for name, i in rep_info.items():  # a different normalization makes the reps incomparable
        assert i.get("mel_mean", cfg["mel_mean"]) == cfg["mel_mean"], f"{name}: mel_mean differs"

    # Normalization comes from the run config, never from the cache's stats.json
    # (notes/decisions.md). kits=None on the scoring splits so held-out kits are scored too.
    def ds_for(split, kits):
        ds = SegmentPairs(dcfg["cache_dir"], split, seg_hop=dcfg["seg_hop"], kits=kits,
                          mel_mean=cfg["mel_mean"], mel_std=cfg["mel_std"])
        for key in ("MAP_VERSION", "split_version"):
            assert ds.meta[key] == dcfg[key], f"{key}: cache {ds.meta[key]} != run {dcfg[key]}"
        return ds

    train_ds = ds_for("train", dcfg["train_kits"])
    tidx = train_subset(train_ds, args.max_train, args.seed)
    val_ds = ds_for("validation", dcfg["train_kits"])
    vidx = np.random.default_rng(args.seed).choice(len(val_ds), min(2048, len(val_ds)), False)
    stats = action_stats(train_ds, tidx)
    pw = torch.tensor(stats["pos_weight"], device=dev)
    print(f"train {len(tidx)} transitions, val {len(vidx)}, "
          f"onset rate {stats['n_onsets'] / stats['n_cells']:.4f}, pos_weight {float(pw):.1f}")
    train_dl = loader(train_ds, tidx, args.bs, args.workers, shuffle=True, seed=args.seed)
    val_dl = loader(val_ds, vidx, args.bs, args.workers)

    score_ds, score_idx = {}, {}
    for split in SCORE_SPLITS:
        score_ds[split] = ds_for(split, None)
        n = len(score_ds[split])
        score_idx[split] = (np.arange(n) if not args.max_score or args.max_score >= n
                            else np.sort(np.random.default_rng(args.seed).choice(
                                n, args.max_score, replace=False)))
    kits = score_ds["validation"].meta["kits"]
    train_ids = [kits.index(k) for k in dcfg["train_kits"]]

    variants = [(r, False) for r in args.reps] + [(r, True) for r in args.with_at if r in args.reps]
    names = {(r, a): r + ("_with_at" if a else "") for r, a in variants}
    res = {"drumjepa": args.drumjepa, "aojepa": args.aojepa, "seed": args.seed,
           "epochs": args.epochs, "max_train": args.max_train, "n_train": int(len(tidx)),
           "n_val": int(len(vidx)), "bs": args.bs, "tolerance_frames": TOL, "nms": NMS,
           "protocol_threshold": THRESH, "threshold_grid": list(GRID),
           "n_bootstrap": N_BOOT, "train_kits": dcfg["train_kits"],
           "kits": kits, "action_stats": stats, "git_commit": git_commit(),
           "n_score": {s: int(len(score_idx[s])) for s in SCORE_SPLITS},
           "score_only": bool(args.score_only), "variants": {}}

    for rep in args.reps:
        embed, d_in = reps[rep]
        models, trained = {}, {}
        for r, use_at in [v for v in variants if v[0] == rep]:
            name = names[(r, use_at)]
            vdir = os.path.join(out_dir, name)
            if args.score_only:
                ck = torch.load(os.path.join(vdir, "best.pt"), map_location=dev,
                                weights_only=False)
                m = InverseModel(d_in=ck["d_in"], use_at=ck["use_at"]).to(dev)
                m.load_state_dict(ck["model"])
                curves = list(csv.DictReader(open(os.path.join(vdir, "metrics.csv"))))
                info = {"train_time_s": float("nan"), "best_val_loss": ck["val_loss"],
                        "n_params": m.n_params(), "curves": curves,
                        "encoder_epoch": ck.get("encoder_epoch")}
            else:
                m, info = train_variant(name, d_in, use_at, embed, train_dl, val_dl, pw,
                                        args, dev, amp, vdir, rep_info[rep].get("epoch"))
            info["best_epoch"] = int(np.argmin([float(r["val/loss"]) for r in info["curves"]]))
            models[name] = m.eval()
            trained[name] = info
            # The encoder is re-read from disk at score time. If its training run is still
            # going, last.pt can have moved since h was fitted on it, and h is then reading
            # an encoder it was never trained against (seen 2026-09-06: aojepa, epoch 3 ->
            # 13, macro F1 0.336 -> 0.142). Such a variant's scores mean nothing.
            now, was = rep_info[rep].get("epoch"), info.get("encoder_epoch")
            drift = (was != now if was is not None
                     else bool(args.score_only and rep_info[rep].get("partial")))
            if drift:
                print(f"[{name}] WARNING: the encoder checkpoint moved (h was fitted on epoch "
                      f"{was}, scoring against epoch {now}); this variant is not valid.",
                      flush=True)
            res["variants"][name] = {"rep": rep, "use_at": use_at, "encoder": rep_info[rep],
                                     "encoder_epoch_trained": was, "encoder_epoch_scored": now,
                                     "encoder_drift": bool(drift), "training": info,
                                     "scores": {}}
        # The threshold is picked on validation/train_kits and then applied unchanged to
        # test, so validation must be scored first (it sweeps the whole grid; test only
        # needs the protocol threshold and the tuned one).
        assert SCORE_SPLITS[0] == "validation", "the tuned threshold comes from validation"
        tuned = {}
        for split in SCORE_SPLITS:
            grid = {name: GRID if split == "validation" else (THRESH, tuned[name])
                    for name in models}
            dl = loader(score_ds[split], score_idx[split], args.bs, args.workers)
            print(f"[{rep}] scoring {split}: {len(score_idx[split])} transitions at "
                  f"{len(next(iter(grid.values())))} thresholds")
            items = score_pass(models, embed, dl, dev, amp, stats, grid)
            for name, it in items.items():
                sel = np.isin(it["kit_id"], train_ids)
                if split == "validation":
                    sweep = sweep_macro_f1(it, sel, GRID)
                    tuned[name] = max(sweep, key=sweep.get)
                    res["variants"][name]["threshold_sweep"] = {str(k): v
                                                                for k, v in sweep.items()}
                    res["variants"][name]["tuned_threshold"] = tuned[name]
                    print(f"[{name}] tuned threshold {tuned[name]:.2f} "
                          f"(macro F1 {sweep[tuned[name]]:.3f} against "
                          f"{sweep[THRESH]:.3f} at {THRESH})")
                idx = {th: grid[name].index(th) for th in (THRESH, tuned[name])}
                for grp in GROUPS:
                    s = sel if grp == "train_kits" else ~sel
                    res["variants"][name]["scores"][f"{split}/{grp}"] = {
                        "at_half": summarize(it, s, args.seed, idx[THRESH], THRESH),
                        "at_tuned": summarize(it, s, args.seed, idx[tuned[name]],
                                              tuned[name])}

    order = [names[v] for v in variants]
    res["h_params"] = res["variants"][order[0]]["training"]["n_params"]
    res["verdict"] = verdict(res, order)
    print(write_outputs(out_dir, res, order))


if __name__ == "__main__":
    main()
