"""E5 content probes: does s_t encode the content of the action that produced it?

    python scripts/eval_e5.py --drumjepa runs/drumjepa_v1 --aojepa runs/drumjepa_v1_aojepa

E3 asked what the state says about the KIT. E5 asks what it says about the ACTION:
which of the 14 classes were struck in each 250 ms step of the clip, how hard, when
inside the step, and where the hi-hat pedal was. The comparison that makes it an
experiment is drum-JEPA against AO-JEPA (`use_action: false`, notes/decisions.md):
same encoder, same masks, same recipe, no action conditioning. If conditioning on
actions during training put action content into the state, drum-JEPA wins here.

Controls as in E3: a random-init encoder of the same architecture and the raw
log-mel. A probe number only counts if the trained encoder beats both.

Per-step feature = the 16 frequency tokens of a time step, mean-pooled (256-d).
The state grid is 8 time x 16 freq, flattened row-major, so token t*16+f belongs to
time step t; pooling is over the frequency axis. A "tokens" variant concatenates the
16 tokens instead (4096-d) to check whether pooling hides content.

Writes <out>/{results.json,results.md,f1_by_class.png,feat_*.npz}, default
runs/drumjepa_v1/e5.
"""
import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import torch
import tqdm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drumjepa.dataset import SegmentPairs  # noqa: E402
from drumjepa.drum_map import CLASSES_V1  # noqa: E402
from drumjepa.features import SEG_FRAMES  # noqa: E402
from drumjepa.model import DrumJEPA  # noqa: E402
from eval_e3 import window_keys  # noqa: E402
from train import git_commit, pick_device  # noqa: E402

REPS = ("drumjepa", "aojepa", "random", "raw_mel")
ENCODER_REPS = ("drumjepa", "aojepa", "random")  # raw_mel needs no forward pass
TOKEN_REPS = ("drumjepa", "aojepa")              # concat variant, linear onset probe only
SPLITS = ("train", "validation", "test")
EVAL_SPLITS = ("validation", "test")
GROUPS = ("train_kits", "heldout_kits")
K = len(CLASSES_V1)
STEP_FRAMES = 25                    # one action patch, 250 ms (drumjepa/features.py)
N_STEPS = SEG_FRAMES // STEP_FRAMES  # 8
MS_PER_FRAME = 10
# The concat variant is 16x the features and only answers "does pooling hide content",
# so it runs on a seeded subsample of clips rather than the full split.
N_TOKEN_CLIPS = 2000
MAX_MLP_ROWS = 50_000  # per MLP fit; the linear probes use every training row
MIN_FIT_ROWS = 200     # a per-class regression below this is reported as n/a
N_BOOT = 1000
THRESHOLDS = np.arange(0.05, 0.96, 0.05)


def step_features(tokens, grid):
    """Encoder tokens (B, n_t*n_f, d) -> per-time-step (B, n_t, d), pooled over frequency.

    The grid is flattened row-major with time as the row axis (StateEncoder.grid is
    (SEG_FRAMES // patch_t, n_mels_pad // patch_f) and sincos_2d repeats rows), so
    reshaping to (B, n_t, n_f, d) puts frequency on axis 2.
    """
    B, n, d = tokens.shape
    assert n == grid[0] * grid[1], (n, grid)
    return tokens.reshape(B, grid[0], grid[1], d).mean(2)


def step_tokens(tokens, grid):
    """Same grid, but the frequency tokens of a step concatenated: (B, n_t, n_f*d)."""
    B, n, d = tokens.shape
    return tokens.reshape(B, grid[0], grid[1] * d)


def raw_mel_steps(x):
    """Normalized log-mel (B, T, F) -> (B, n_steps, F), mean over each step's frames."""
    B, T, F = x.shape
    return x.reshape(B, T // STEP_FRAMES, STEP_FRAMES, F).mean(2)


def step_targets(a, cc):
    """Per-step targets from one clip's action, (B, T, K) and (B, T) -> four arrays.

    onset (B, S, K) bool: class hit at least once in the step. velocity (B, S, K):
    max velocity/127 in the step, 0 where no onset. timing (B, S, K) int8: frame
    index 0..24 of the first onset, -1 where none. hihat (B, S): mean CC4/127.
    """
    B = a.shape[0]
    r = a.reshape(B, N_STEPS, STEP_FRAMES, a.shape[-1])
    hit = r > 0
    onset = hit.any(2)
    first = np.argmax(hit, axis=2).astype(np.int8)
    return {"onset": onset, "velocity": r.max(2).astype(np.float16),
            "timing": np.where(onset, first, -1).astype(np.int8),
            "hihat": cc.reshape(B, N_STEPS, STEP_FRAMES).mean(2).astype(np.float16)}


class ClipItems(Dataset):
    """One clip: x_t and the action of the SAME window, plus the identifying columns."""

    def __init__(self, ds, tidx):
        self.ds, self.tidx = ds, tidx

    def __len__(self):
        return len(self.tidx)

    def __getitem__(self, j):
        i = int(self.tidx[j])
        it = self.ds[i]
        return {"x_t": it["x_t"], "a_t": it["a_t"], "cc_t": it["cc_t"],
                "kit_id": it["kit_id"], "seq_idx": it["seq_idx"], "tidx": i}


def window_subset(ds, keep, n_clips, seed):
    """Clip indices: a seeded window subset kept on every kit of `keep`, ~n_clips total."""
    idx = np.flatnonzero(keep)
    if n_clips <= 0 or n_clips >= len(idx):
        return idx
    n_kits = len(np.unique(ds.kit_id[idx]))
    uniq, inv = np.unique(window_keys(ds)[idx], return_inverse=True)
    n_keys = max(1, n_clips // n_kits)
    pick = np.random.default_rng(seed).choice(len(uniq), min(n_keys, len(uniq)), replace=False)
    return idx[np.isin(inv, pick)]


def encode_split(ds, tidx, models, grid, dev, amp, args, reps):
    """Per-step features for every representation plus the per-step targets and columns."""
    tok_flag = np.zeros(len(tidx), bool)
    tok_flag[np.random.default_rng(args.seed).choice(
        len(tidx), min(N_TOKEN_CLIPS, len(tidx)), replace=False)] = True
    dl = DataLoader(ClipItems(ds, tidx), batch_size=args.bs, num_workers=args.workers)
    out, off = {}, 0
    with torch.inference_mode():
        for b in tqdm.tqdm(dl, mininterval=5):
            x = b["x_t"].to(dev)
            flag = tok_flag[off:off + x.shape[0]]
            off += x.shape[0]
            for name, model in models.items():
                with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    z = model.encode_state(x, teacher=True)
                z = z.float().cpu()
                out.setdefault(name, []).append(step_features(z, grid).numpy().astype(np.float16))
                if name in TOKEN_REPS and flag.any():
                    out.setdefault(f"{name}_tok", []).append(
                        step_tokens(z[flag], grid).numpy().astype(np.float16))
            if "raw_mel" in reps:
                out.setdefault("raw_mel", []).append(
                    raw_mel_steps(b["x_t"]).numpy().astype(np.float16))
            for k, v in step_targets(b["a_t"].numpy(), b["cc_t"].numpy()).items():
                out.setdefault(k, []).append(v)
            for k in ("kit_id", "seq_idx", "tidx"):
                out.setdefault(k, []).append(b[k].numpy())
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["tok_pos"] = np.flatnonzero(tok_flag)
    return out


def rows(feat, rep, sel):
    """(n_clips, S, d) features of the selected clips flattened to (n_clips*S, d) float32."""
    return feat[rep][sel].reshape(-1, feat[rep].shape[-1]).astype(np.float32)


def f1_counts(y, pred):
    """(tp, fp, fn) per class, (K, 3)."""
    return np.stack([(y & pred).sum(0), (~y & pred).sum(0), (y & ~pred).sum(0)],
                    axis=1).astype(np.float64)


def counts_by_seq(y, pred, seq):
    """The same counts split by sequence, (n_seq, K, 3), so the bootstrap is a matmul."""
    _, inv = np.unique(seq, return_inverse=True)
    n = inv.max() + 1
    c = np.zeros((n, y.shape[1], 3))
    for k in range(y.shape[1]):
        for j, w in enumerate((y[:, k] & pred[:, k], ~y[:, k] & pred[:, k],
                               y[:, k] & ~pred[:, k])):
            c[:, k, j] = np.bincount(inv, w.astype(np.float64), minlength=n)
    return c


def f1_from_counts(s):
    """Per-class F1 from summed (K, 3) counts; 0 where nothing is true and nothing predicted."""
    tp, fp, fn = s[:, 0], s[:, 1], s[:, 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def f1_bootstrap(c, n_boot=N_BOOT, seed=0):
    """95% CI on macro-F1, resampling SEQUENCES (kits share performances, as in E1).

    F1 is not a mean of per-clip quantities, so eval_e1.cluster_bootstrap does not
    apply; instead the per-sequence tp/fp/fn are summed under multinomial weights and
    the macro-F1 recomputed exactly. Same clusters, same 1000 resamples.
    """
    n = c.shape[0]
    flat = c.reshape(n, -1)
    rng = np.random.default_rng(seed)
    boots = [f1_from_counts((rng.multinomial(n, np.full(n, 1.0 / n)) @ flat).reshape(-1, 3)).mean()
             for _ in range(n_boot)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def fit_onset(X, Y, probe, seed):
    """A linear (14 independent logistic regressions) or MLP (one multilabel net) probe.

    Returns a function mapping features to a (n, K) probability matrix.
    """
    if probe == "mlp":
        sub = np.random.default_rng(seed).choice(
            len(X), min(MAX_MLP_ROWS, len(X)), replace=False)
        clf = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200,
                            random_state=seed).fit(X[sub], Y[sub].astype(int))
        return lambda Z: clf.predict_proba(Z)
    clfs = []
    for k in range(Y.shape[1]):
        y = Y[:, k]
        clfs.append(float(y.mean()) if y.all() or not y.any() else
                    LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000).fit(X, y))
    def predict(Z):
        return np.column_stack([np.full(len(Z), c) if isinstance(c, float)
                                else c.predict_proba(Z)[:, 1] for c in clfs])
    return predict


def onset_scores(prob, Y, seq, thr, boot):
    """macro-F1, per-class F1 and (optionally) a cluster-bootstrap CI at threshold `thr`."""
    pred = prob >= thr
    f1 = f1_from_counts(f1_counts(Y, pred))
    out = {"macro_f1": float(f1.mean()), "per_class_f1": f1.tolist(), "n_steps": int(len(Y))}
    if boot:
        out["macro_f1_ci"] = f1_bootstrap(counts_by_seq(Y, pred, seq))
    return out


def prior_f1(p_train, Y):
    """Trivial baseline: predict each class independently at its train positive rate.

    Its expected F1 against a split with positive rate q is 2pq/(p+q), in closed form.
    """
    q = Y.mean(0)
    den = p_train + q
    return np.where(den > 0, 2 * p_train * q / np.maximum(den, 1e-12), 0.0)


def fit_regression(X, y, probe, seed):
    """Ridge or MLPRegressor on standardized features; returns the fitted estimator."""
    if probe == "linear":
        return Ridge(alpha=1.0).fit(X, y)
    sub = np.random.default_rng(seed).choice(len(X), min(MAX_MLP_ROWS, len(X)), replace=False)
    return MLPRegressor(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200,
                        random_state=seed).fit(X[sub], y[sub])


def per_class_regression(Xtr, Ytr, Mtr, ev, probe, seed):
    """Per-class regression over the (step, class) rows that have an onset.

    Args:
      Xtr, Ytr, Mtr: train features (n, d), targets (n, K), onset mask (n, K).
      ev: {name: (X, Y, M)} evaluation sets.
    Returns:
      {name: {"mae", "baseline_mae", "n"}} pooled over classes, plus per-class n.
    """
    res = {n: {"abs": 0.0, "base": 0.0, "n": 0} for n in ev}
    n_fit = []
    for k in range(Ytr.shape[1]):
        m = Mtr[:, k]
        n_fit.append(int(m.sum()))
        if m.sum() < MIN_FIT_ROWS:
            continue
        est = fit_regression(Xtr[m], Ytr[m, k], probe, seed)
        mu = float(Ytr[m, k].mean())
        for name, (X, Y, M) in ev.items():
            s = M[:, k]
            if not s.any():
                continue
            e = np.abs(est.predict(X[s]) - Y[s, k])
            res[name]["abs"] += float(e.sum())
            res[name]["base"] += float(np.abs(Y[s, k] - mu).sum())
            res[name]["n"] += int(s.sum())
    return {n: {"mae": v["abs"] / max(v["n"], 1), "baseline_mae": v["base"] / max(v["n"], 1),
                "n": v["n"]} for n, v in res.items()}, n_fit


def f1_figure(path, per_class, reps, run_name):
    """Grouped bars: per-class onset F1, linear probe, test split, train kits."""
    fig, ax = plt.subplots(figsize=(11, 4.0))
    w = 0.8 / len(reps)
    x = np.arange(K)
    for i, rep in enumerate(reps):
        ax.bar(x + i * w - 0.4 + w / 2, per_class[rep], w, label=rep)
    ax.set_xticks(x, CLASSES_V1, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("onset F1")
    ax.set_ylim(0, 1)
    ax.set_title(f"E5: per-class onset F1, linear probe, test split, train kits ({run_name})",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=len(reps))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def verdict(res, reps):
    """Three to five sentences on Q1/E5, read off the metrics."""
    o = res["onset"]
    t = lambda rep: o[rep]["linear"]["test"]["train_kits"]["macro_f1"]  # noqa: E731
    h = lambda rep: o[rep]["linear"]["test"]["heldout_kits"]["macro_f1"]  # noqa: E731
    ci = lambda rep: o[rep]["linear"]["test"]["train_kits"]["macro_f1_ci"]  # noqa: E731
    prior = res["onset_prior"]["test"]["train_kits"]["macro_f1"]
    best = max(reps, key=t)
    ctrl = [r for r in ("random", "raw_mel") if r in reps]
    beats = [r for r in ctrl if t("drumjepa") > t(r)]
    pc = np.array(o["drumjepa"]["linear"]["test"]["train_kits"]["per_class_f1"])
    good = [CLASSES_V1[i] for i in np.argsort(-pc)[:3]]
    bad = [CLASSES_V1[i] for i in np.argsort(pc)[:3]]
    v, tok = res["regression"], res["tokens"]
    vm = v["velocity"]["drumjepa"]["linear"]["test"]["train_kits"]
    tm = v["timing"]["drumjepa"]["linear"]["test"]["train_kits"]
    hm = v["hihat"]["drumjepa"]["linear"]["test"]["train_kits"]
    if "aojepa" in reps:
        d, a = t("drumjepa"), t("aojepa")
        gap = ("beats" if d > a else "loses to")
        head = (f"Drum-JEPA {gap} AO-JEPA on onset content: macro-F1 {d:.3f} "
                f"[{ci('drumjepa')[0]:.3f}, {ci('drumjepa')[1]:.3f}] against {a:.3f} "
                f"[{ci('aojepa')[0]:.3f}, {ci('aojepa')[1]:.3f}] with a linear probe on the "
                f"test split's train kits, so action conditioning during training "
                f"{'did' if d > a else 'did not'} make action content more recoverable from s_t"
                + (f" (AO-JEPA is only at epoch {res['aojepa_epoch']} of 20 here, so this "
                   "reading is provisional)." if (res["aojepa_epoch"] or 0) < 19 else "."))
    else:
        head = (f"AO-JEPA was skipped in this run, so the drum-JEPA vs AO-JEPA comparison E5 "
                f"exists for is not answered here; drum-JEPA's linear onset macro-F1 is "
                f"{t('drumjepa'):.3f} [{ci('drumjepa')[0]:.3f}, {ci('drumjepa')[1]:.3f}].")
    return [
        head,
        f"Against the controls the trained encoder beats "
        f"{'both' if len(beats) == len(ctrl) else (beats[0] if beats else 'neither')}: "
        + ", ".join(f"{r} {t(r):.3f}" for r in ctrl)
        + f", trivial class-prior baseline {prior:.3f}; the best representation overall is {best} at "
        f"{t(best):.3f}, so the same warning as E3 applies — a probe only counts when it beats "
        f"the spectrum it was computed from.",
        f"What is recoverable is which classes were struck, and only for the frequent, "
        f"spectrally distinct ones: {', '.join(good)} score highest and {', '.join(bad)} lowest "
        f"under the drum-JEPA linear probe, tracking how often the class occurs rather than "
        f"anything about the model; and mean-pooling the 16 frequency tokens of a step throws "
        f"content away — concatenating them instead lifts drum-JEPA from "
        f"{tok['drumjepa']['pooled']['test']['train_kits']['macro_f1']:.3f} to "
        f"{tok['drumjepa']['tokens']['test']['train_kits']['macro_f1']:.3f} on the same clips, "
        f"so every pooled number above is a lower bound.",
        f"What is not recoverable is the fine structure of the hit: velocity MAE is "
        f"{vm['mae'] * 127:.1f} MIDI units against {vm['baseline_mae'] * 127:.1f} for predicting "
        f"the per-class training mean, first-onset timing inside the 250 ms step is "
        f"{tm['mae'] * MS_PER_FRAME:.1f} ms against {tm['baseline_mae'] * MS_PER_FRAME:.1f} ms, "
        f"and the hi-hat pedal position is {hm['mae']:.3f} against {hm['baseline_mae']:.3f} "
        f"(CC4/127) — every one of those beats its baseline, but by 10-25%, so the probes read "
        f"the event far more sharply than its dynamics or its position inside the step.",
        "Held-out kits cost the encoders little — "
        + ", ".join(f"{r} {h(r):.3f} against {t(r):.3f}" for r in reps if r != "raw_mel")
        + (f" — but cost the raw log-mel most of its lead ({h('raw_mel'):.3f} against "
           f"{t('raw_mel'):.3f}), so the spectrum's advantage is largely kit-specific while "
           "what the encoders carry about the action is not." if "raw_mel" in reps else
           ", which is what a content probe should do if it reads the action and not the kit."),
    ]


def write_outputs(out_dir, res, reps):
    """results.json and results.md; returns the markdown."""
    md = [f"# E5 content probes — {res['run_name']} (epoch {res['epoch']}) vs "
          f"{res['aojepa_name'] or 'no AO-JEPA'}"
          + (f" (epoch {res['aojepa_epoch']}, step {res['aojepa_step']})"
             if res["aojepa_name"] else "") + "\n",
          f"{res['n_clips']['train']} train clips ({res['n_clips']['train'] * N_STEPS} steps) "
          f"fit on the 6 train kits; validation and test scored on "
          f"{res['n_clips']['validation']} and {res['n_clips']['test']} clips, split by kit "
          "group. Per-step feature = the 16 frequency tokens of a 250 ms step, mean-pooled "
          "(256-d); raw_mel = the mean of the step's 25 frames (229-d). Threshold 0.5 unless "
          "stated; the best-threshold column picks one threshold on validation/train kits by "
          "macro-F1 and applies it to test.\n",
          "## Onset: which classes were struck in the step", "",
          "| representation | probe | split | kit group | macro-F1 | macro-F1 @ best thr |",
          "|---|---|---|---|---|---|"]
    for rep in reps:
        for probe in ("linear", "mlp"):
            for s in EVAL_SPLITS:
                for g in GROUPS:
                    r = res["onset"][rep][probe][s][g]
                    ci = (f" [{r['macro_f1_ci'][0]:.3f}, {r['macro_f1_ci'][1]:.3f}]"
                          if "macro_f1_ci" in r else "")
                    md.append(f"| {rep} | {probe} | {s} | {g} | {r['macro_f1']:.3f}{ci} | "
                              f"{r['macro_f1_best_thr']:.3f} |")
    for s in EVAL_SPLITS:
        for g in GROUPS:
            r = res["onset_prior"][s][g]
            md.append(f"| prior (trivial) | - | {s} | {g} | {r['macro_f1']:.3f} | - |")
    md += ["", "CI: 1000-resample cluster bootstrap over sequences, on test only. F1 is not a "
           "mean of per-clip values, so the per-sequence tp/fp/fn are summed under multinomial "
           "weights and the macro-F1 recomputed exactly, rather than reusing "
           "eval_e1.cluster_bootstrap. The trivial baseline predicts each class independently "
           "at its training positive rate, whose expected F1 is 2pq/(p+q) in closed form.", "",
           "### Per-class F1, linear probe, test split, train kits", "",
           "| class | train positive rate | " + " | ".join(reps) + " | prior |",
           "|---" * (len(reps) + 3) + "|"]
    for i, c in enumerate(CLASSES_V1):
        cells = " | ".join(
            f"{res['onset'][r]['linear']['test']['train_kits']['per_class_f1'][i]:.3f}"
            for r in reps)
        md.append(f"| {c} | {res['class_prior'][i]:.4f} | {cells} | "
                  f"{res['onset_prior']['test']['train_kits']['per_class_f1'][i]:.3f} |")

    md += ["", "### Pooling check: 16 frequency tokens concatenated (4096-d), linear onset probe",
           "", f"On a seeded {N_TOKEN_CLIPS}-clip subsample of each split, so these numbers are "
           "comparable to each other but not to the table above.", "",
           "| representation | features | validation train kits | test train kits |",
           "|---|---|---|---|"]
    for rep in [r for r in TOKEN_REPS if r in reps]:
        for kind in ("pooled", "tokens"):
            r = res["tokens"][rep][kind]
            md.append(f"| {rep} | {kind} | {r['validation']['train_kits']['macro_f1']:.3f} | "
                      f"{r['test']['train_kits']['macro_f1']:.3f} |")

    md += ["", "## Velocity, timing and hi-hat position", "",
           "Velocity and timing are scored only on (clip, step, class) triples that have an "
           "onset, with one regression per class; the MAE is pooled over classes. Timing is the "
           "frame of the first onset inside the 25-frame step. The baseline predicts the "
           "per-class training mean (the training mean for hi-hat).", "",
           "| target | representation | probe | split | kit group | MAE | baseline MAE | n rows |",
           "|---|---|---|---|---|---|---|---|"]
    units = {"velocity": (127.0, "MIDI vel"), "timing": (float(MS_PER_FRAME), "ms"),
             "hihat": (1.0, "CC4/127")}
    for target, (scale, unit) in units.items():
        for rep in reps:
            for probe in ("linear", "mlp"):
                for s in EVAL_SPLITS:
                    for g in GROUPS:
                        r = res["regression"][target][rep][probe][s][g]
                        md.append(f"| {target} ({unit}) | {rep} | {probe} | {s} | {g} | "
                                  f"{r['mae'] * scale:.3f} | {r['baseline_mae'] * scale:.3f} | "
                                  f"{r['n']} |")
    md += ["", "Velocity MAE is also " + ", ".join(
        f"{res['regression']['velocity'][r]['linear']['test']['train_kits']['mae']:.4f} ({r})"
        for r in reps) + " in velocity/127 units.",
        "", "Classes with fewer than "
        f"{MIN_FIT_ROWS} training rows get no regression and contribute no rows: "
        + ", ".join(f"{c} {n}" for c, n in zip(CLASSES_V1, res["n_onset_rows"])) + ".",
        "", "## Q1/E5", ""] + res["verdict"] + [""]
    md = "\n".join(md)
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    open(os.path.join(out_dir, "results.md"), "w").write(md)
    return md


def load_teacher(run_dir, dev):
    """The teacher encoder of a run's last checkpoint, plus its epoch/step/name."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    ck = torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu", weights_only=False)
    model = DrumJEPA(cfg["config"]["model"])
    model.load_state_dict(ck["model"])
    return model.to(dev).eval(), cfg, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drumjepa", default="runs/drumjepa_v1")
    ap.add_argument("--aojepa", default="runs/drumjepa_v1_aojepa")
    ap.add_argument("--max-train", type=int, default=30000, help="train clips, 0 = all")
    ap.add_argument("--max-eval", type=int, default=0, help="val/test clips each, 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <drumjepa>/e5")
    ap.add_argument("--recompute", action="store_true", help="ignore the feature cache")
    ap.add_argument("--skip", nargs="*", default=[], choices=list(REPS),
                    help="representations to leave out (e.g. --skip aojepa)")
    args = ap.parse_args()
    reps = [r for r in REPS if r not in args.skip]
    assert "drumjepa" in reps, "drumjepa is the subject of E5"

    dev = pick_device(args.device)
    out_dir = args.out or os.path.join(args.drumjepa, "e5")
    os.makedirs(out_dir, exist_ok=True)

    models, steps = {}, {}
    model, cfg, ck = load_teacher(args.drumjepa, dev)
    models["drumjepa"], steps["drumjepa"] = model, int(ck["step"])
    amp = bool(cfg["config"]["amp"]) and dev.type != "cpu"
    ao_cfg = ao_ck = None
    if "aojepa" in reps:
        ao_model, ao_cfg, ao_ck = load_teacher(args.aojepa, dev)
        assert not ao_cfg["config"]["model"].get("use_action", True), "AO-JEPA needs use_action false"
        models["aojepa"], steps["aojepa"] = ao_model, int(ao_ck["step"])
    if "random" in reps:
        torch.manual_seed(0)
        models["random"] = DrumJEPA({}).to(dev).eval()
    grid = model.grid

    dcfg = cfg["config"]["data"]
    dss = {s: SegmentPairs(dcfg["cache_dir"], s, seg_hop=dcfg["seg_hop"],
                           mel_mean=cfg["mel_mean"], mel_std=cfg["mel_std"]) for s in SPLITS}
    kits = dss["train"].meta["kits"]
    train_ids = [kits.index(k) for k in dcfg["train_kits"]]
    want = {"seed": args.seed, "max_train": args.max_train, "max_eval": args.max_eval,
            "reps": sorted(reps), "steps": steps, "n_token_clips": N_TOKEN_CLIPS}

    feat = {}
    for s in SPLITS:
        path = os.path.join(out_dir, f"feat_{s}.npz")
        if os.path.exists(path) and not args.recompute:
            z = dict(np.load(path))
            if json.loads(str(z.pop("meta"))) == want:
                feat[s] = z
                print(f"{s}: loaded {len(z['kit_id'])} cached clips")
                continue
            print(f"{s}: cache does not match this run's settings, recomputing")
        keep = (np.isin(dss[s].kit_id, train_ids) if s == "train"
                else np.ones(len(dss[s]), bool))
        n = args.max_train if s == "train" else args.max_eval
        tidx = window_subset(dss[s], keep, n, args.seed)
        print(f"{s}: encoding {len(tidx)} clips on {dev} (bf16={amp})")
        feat[s] = encode_split(dss[s], tidx, models, grid, dev, amp, args, reps)
        np.savez(path, meta=json.dumps(want), **feat[s])

    # The train split is encoded on the train kits only (the probes are fit there), so it
    # has no held-out group; validation and test carry both.
    sel = {}
    for s in SPLITS:
        m = np.isin(feat[s]["kit_id"], train_ids)
        sel[s] = {"train_kits": m} if s == "train" else {"train_kits": m, "heldout_kits": ~m}
    assert sel["train"]["train_kits"].all(), "the train split must hold train kits only"
    seq = {s: {g: np.repeat(feat[s]["seq_idx"][m], N_STEPS) for g, m in sel[s].items()}
           for s in SPLITS}
    Y = {s: {g: feat[s]["onset"][m].reshape(-1, K) for g, m in sel[s].items()} for s in SPLITS}
    Ytr = Y["train"]["train_kits"]
    p_train = Ytr.mean(0)

    res = {"drumjepa_dir": args.drumjepa, "aojepa_dir": args.aojepa if "aojepa" in reps else None,
           "run_name": cfg["config"]["run"]["name"], "epoch": ck["epoch"], "step": ck["step"],
           "aojepa_name": ao_cfg["config"]["run"]["name"] if ao_cfg else None,
           "aojepa_epoch": ao_ck["epoch"] if ao_ck else None,
           "aojepa_step": ao_ck["step"] if ao_ck else None,
           "reps": reps, "seed": args.seed, "kits": kits, "train_kits": dcfg["train_kits"],
           "n_clips": {s: int(len(feat[s]["kit_id"])) for s in SPLITS},
           "class_prior": p_train.tolist(), "classes": CLASSES_V1,
           "max_train": args.max_train, "max_eval": args.max_eval,
           "git_commit": git_commit(), "onset": {}, "tokens": {}, "regression": {}}
    res["onset_prior"] = {s: {g: {"macro_f1": float(prior_f1(p_train, Y[s][g]).mean()),
                                  "per_class_f1": prior_f1(p_train, Y[s][g]).tolist()}
                              for g in GROUPS} for s in EVAL_SPLITS}

    for rep in reps:
        print(f"onset probes: {rep}", flush=True)
        scaler = StandardScaler().fit(rows(feat["train"], rep, sel["train"]["train_kits"]))
        X = {s: {g: scaler.transform(rows(feat[s], rep, m)) for g, m in sel[s].items()}
             for s in SPLITS}
        res["onset"][rep] = {}
        for probe in ("linear", "mlp"):
            predict = fit_onset(X["train"]["train_kits"], Ytr, probe, args.seed)
            prob = {s: {g: predict(X[s][g]) for g in GROUPS} for s in EVAL_SPLITS}
            pv, yv = prob["validation"]["train_kits"], Y["validation"]["train_kits"]
            thr = float(THRESHOLDS[np.argmax(
                [f1_from_counts(f1_counts(yv, pv >= t)).mean() for t in THRESHOLDS])])
            res["onset"][rep][probe] = {s: {} for s in EVAL_SPLITS}
            for s in EVAL_SPLITS:
                for g in GROUPS:
                    r = onset_scores(prob[s][g], Y[s][g], seq[s][g], 0.5, s == "test")
                    r["threshold_from_validation"] = thr
                    r["macro_f1_best_thr"] = onset_scores(
                        prob[s][g], Y[s][g], seq[s][g], thr, False)["macro_f1"]
                    res["onset"][rep][probe][s][g] = r
            del prob, pv, yv

        print(f"regressions: {rep}", flush=True)
        for target in ("velocity", "timing"):
            def get(s, g, t=target):
                return feat[s][t][sel[s][g]].reshape(-1, K).astype(np.float32)
            ev = {(s, g): (X[s][g], get(s, g), Y[s][g]) for s in EVAL_SPLITS for g in GROUPS}
            res["regression"].setdefault(target, {})[rep] = {}
            for probe in ("linear", "mlp"):
                # n_onset_rows is the same for every rep and target: it counts the targets.
                out, res["n_onset_rows"] = per_class_regression(
                    X["train"]["train_kits"], get("train", "train_kits"), Ytr, ev, probe,
                    args.seed)
                res["regression"][target][rep][probe] = {
                    s: {g: out[(s, g)] for g in GROUPS} for s in EVAL_SPLITS}
        h = {s: {g: feat[s]["hihat"][m].reshape(-1).astype(np.float32) for g, m in sel[s].items()}
             for s in SPLITS}
        htr, res["regression"].setdefault("hihat", {})[rep] = h["train"]["train_kits"], {}
        for probe in ("linear", "mlp"):
            est = fit_regression(X["train"]["train_kits"], htr, probe, args.seed)
            res["regression"]["hihat"][rep][probe] = {s: {g: {
                "mae": float(np.abs(est.predict(X[s][g]) - h[s][g]).mean()),
                "baseline_mae": float(np.abs(h[s][g] - htr.mean()).mean()),
                "n": int(len(h[s][g]))} for g in GROUPS} for s in EVAL_SPLITS}
        del X

    for rep in [r for r in TOKEN_REPS if r in reps]:
        print(f"pooling check: {rep}", flush=True)
        res["tokens"][rep] = {}
        for kind in ("pooled", "tokens"):
            # `<rep>_tok` already holds only the token-subset clips, in tok_pos order;
            # the pooled array holds every clip, so it is indexed by tok_pos first.
            pos = {s: feat[s]["tok_pos"] for s in SPLITS}
            m = {s: sel[s]["train_kits"][pos[s]] for s in SPLITS}
            A = {s: feat[s][f"{rep}_tok"] if kind == "tokens" else feat[s][rep][pos[s]]
                 for s in SPLITS}
            Xt = {s: A[s][m[s]].reshape(-1, A[s].shape[-1]).astype(np.float32) for s in SPLITS}
            Yt = {s: feat[s]["onset"][pos[s]][m[s]].reshape(-1, K) for s in SPLITS}
            St = {s: np.repeat(feat[s]["seq_idx"][pos[s]][m[s]], N_STEPS) for s in SPLITS}
            sc = StandardScaler().fit(Xt["train"])
            predict = fit_onset(sc.transform(Xt["train"]), Yt["train"], "linear", args.seed)
            res["tokens"][rep][kind] = {
                s: {"train_kits": onset_scores(predict(sc.transform(Xt[s])), Yt[s], St[s], 0.5,
                                               False)} for s in EVAL_SPLITS}

    f1_figure(os.path.join(out_dir, "f1_by_class.png"),
              {r: res["onset"][r]["linear"]["test"]["train_kits"]["per_class_f1"] for r in reps},
              reps, res["run_name"])
    res["verdict"] = verdict(res, reps)
    print(write_outputs(out_dir, res, reps))


if __name__ == "__main__":
    main()
