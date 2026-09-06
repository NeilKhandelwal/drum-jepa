"""E3 kit latent: is kit identity readable from the state embedding?

    python scripts/eval_e3.py --run-dir runs/drumjepa_v1

E2 showed the model USES kit identity (a wrong-kit s_t costs 0.02 of prediction
error). Q2 (CLAUDE.md) asks whether it is READABLE: a probe recovers the kit from
s_t, the counterfactual "same window, other kit" displacement has a consistent
direction, and the 8 kits the model never heard embed sensibly.

A clip embedding is the mean over the 128 teacher state tokens (256-d). Three
controls bound it: the student encoder, a random-init encoder with the same
architecture, and the mean log-mel of the clip. A probe number only counts if the
trained teacher beats the last two (notes/decisions.md, "E3 representation").

Writes <out>/{results.json,results.md,confusion.png,centroids.png,emb_*.npz},
default <run-dir>/e3.
"""
import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import torch
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drumjepa.dataset import SegmentPairs  # noqa: E402
from drumjepa.model import DrumJEPA  # noqa: E402
from eval_e1 import cluster_bootstrap, masks_for  # noqa: E402
from train import git_commit, pick_device  # noqa: E402

REPS = ("teacher", "student", "random", "raw_mel")
# Geometry is run on the trained teacher and its two controls; the student tracks the
# teacher too closely on the probes to be worth a fourth pass.
GEOM_REPS = ("teacher", "random", "raw_mel")
SPLITS = ("train", "validation", "test")
COLS = ("kit_id", "seq_idx", "offset", "density", "tidx")
# Windows used for the delta-consistency bootstrap. All 2034 validation windows are
# used for kit-vector transfer; the consistency statistic needs a (182, W, 256)
# array resampled 1000 times, so it is capped here rather than by a flag.
N_GEOM_WINDOWS = 512
N_BOOT = 1000
# Expected nearest train kit for each held-out kit, from the annotations in
# configs/kits_v1.yaml. Cassette (Lo-Fi Compress) counts as acoustic: it is an
# acoustic kit through a lo-fi chain, not a synthesized one.
ACOUSTIC = ("Jazz", "60s Rock", "Studio (Live Room)", "Cassette (Lo-Fi Compress)")
ELECTRONIC = ("Ele-Drum", "808 Simple")
EXPECT = {"Bigga Bop (Jazz)": ("Jazz",), "Pop-Rock (Studio)": ("Studio (Live Room)",),
          "909 Simple": ("808 Simple",), "Acoustic Kit": ACOUSTIC, "Unplugged": ACOUSTIC,
          "Heavy Metal": ACOUSTIC, "Deep Daft": ELECTRONIC, "Raw Dnb (Layered Hybrid)": ELECTRONIC}


def mean_pool_embed(model, x, teacher=True):
    """Clip embedding: mean over the state encoder's tokens, (B, T, N_MELS) -> (B, d)."""
    return model.encode_state(x, teacher=teacher).float().mean(1)


def raw_mel_embed(x):
    """Raw baseline: mean over the frames of the normalized log-mel, (B, T, F) -> (B, F)."""
    return x.float().mean(1)


class ClipItems(Dataset):
    """One clip: the x_t of a transition plus the columns that identify it."""

    def __init__(self, ds, tidx):
        self.ds, self.tidx = ds, tidx

    def __len__(self):
        return len(self.tidx)

    def __getitem__(self, j):
        i = int(self.tidx[j])
        it = self.ds[i]
        return {"x_t": it["x_t"], "kit_id": it["kit_id"], "seq_idx": it["seq_idx"],
                "offset": int(self.ds.offset[i]), "density": it["density_bin"], "tidx": i}


class SwapItems(Dataset):
    """One counterfactual window: s_t and x_{t+1} on both kits, and the shared action.

    The action is identical on every kit (it is built from the canonical MIDI), so
    a_t1/cc_t1 are taken from the kit-B item.
    """

    def __init__(self, ds, i_b, i_a):
        self.ds, self.i_b, self.i_a = ds, i_b, i_a

    def __len__(self):
        return len(self.i_b)

    def __getitem__(self, j):
        b, a = self.ds[int(self.i_b[j])], self.ds[int(self.i_a[j])]
        return {"x_t_B": b["x_t"], "x_t_A": a["x_t"], "a_t1": b["a_t1"], "cc_t1": b["cc_t1"],
                "x_t1_B": b["x_t1"], "x_t1_A": a["x_t1"], "tidx": int(self.i_b[j]),
                "seq_idx": b["seq_idx"], "kit_A": a["kit_id"], "kit_B": b["kit_id"]}


def window_keys(ds):
    """Integer key per transition identifying its (seq_idx, offset) window."""
    return ds.seq_idx.astype(np.int64) * (int(ds.offset.max()) + 1) + ds.offset.astype(np.int64)


def train_subset(ds, max_train, seed):
    """Indices of a seeded window subset kept on every kit, so kit counts stay equal."""
    if max_train <= 0 or max_train >= len(ds):
        return np.arange(len(ds))
    key = window_keys(ds)
    uniq, inv = np.unique(key, return_inverse=True)
    n_keys = max(1, max_train // len(np.unique(ds.kit_id)))
    pick = np.random.default_rng(seed).choice(len(uniq), min(n_keys, len(uniq)), replace=False)
    return np.flatnonzero(np.isin(inv, pick))


def encode_split(ds, tidx, models, dev, amp, bs, workers):
    """All four representations for `tidx`, as a dict of arrays keyed by REPS + COLS."""
    dl = DataLoader(ClipItems(ds, tidx), batch_size=bs, num_workers=workers)
    out = {k: [] for k in REPS + COLS}
    with torch.inference_mode():
        for b in tqdm.tqdm(dl, mininterval=5):
            x = b["x_t"].to(dev)
            for name, (model, teacher) in models.items():
                with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    z = mean_pool_embed(model, x, teacher)
                out[name].append(z.float().cpu().numpy())
            out["raw_mel"].append(raw_mel_embed(b["x_t"]).numpy())
            for k in COLS:
                out[k].append(b[k].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def probe_scores(emb, task_kits, seed):
    """Accuracy of a linear and an MLP probe per representation, plus the fitted pair.

    Trains on the train split restricted to `task_kits` and scores validation and test.
    """
    res, fitted = {}, {}
    for rep in REPS:
        res[rep] = {}
        sel = {s: np.isin(emb[s]["kit_id"], task_kits) for s in SPLITS}
        scaler = StandardScaler().fit(emb["train"][rep][sel["train"]])
        X = {s: scaler.transform(emb[s][rep][sel[s]]) for s in SPLITS}
        y = {s: emb[s]["kit_id"][sel[s]] for s in SPLITS}
        for probe in ("linear", "mlp"):
            clf = (LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000)
                   if probe == "linear" else
                   MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True,
                                 max_iter=300, random_state=0))
            clf.fit(X["train"], y["train"])
            res[rep][probe] = {s: float((clf.predict(X[s]) == y[s]).mean())
                               for s in ("validation", "test")}
            fitted[(rep, probe)] = (scaler, clf)
    return res, fitted


def kit_centroids(e, rep, kit_ids):
    """(n_kits, d) mean embedding per kit for one representation."""
    return np.stack([e[rep][e["kit_id"] == k].mean(0) for k in kit_ids])


def aligned_embeddings(e, kit_ids, rep="teacher"):
    """(n_kits, W, d) embeddings of the windows present on every kit, one representation."""
    key = e["seq_idx"].astype(np.int64) * (int(e["offset"].max()) + 1) + e["offset"]
    uniq, inv = np.unique(key, return_inverse=True)
    pos = {k: np.full(len(uniq), -1, np.int64) for k in kit_ids}
    for i, (k, w) in enumerate(zip(e["kit_id"], inv)):
        if k in pos:
            pos[k][w] = i
    full = np.all([pos[k] >= 0 for k in kit_ids], axis=0)
    return np.stack([e[rep][pos[k][full]] for k in kit_ids])


def delta_geometry(E, pairs, n_boot=N_BOOT, seed=0):
    """Consistency of the kit displacement e_B - e_A across windows.

    Within-pair: mean cosine between the deltas of two different windows of the same
    ordered kit pair, which equals ||M_p||^2 for M_p the mean unit delta of pair p.
    Between-pair: the same quantity across pairs (A,B), (C,D) with {A,B} disjoint
    from {C,D}, i.e. mean |M_p . M_q| — like for like, both attenuated by the same
    per-window noise. The absolute value is necessary: (D,C) is in the pair list
    whenever (C,D) is and carries the negated delta, so the signed mean is
    identically zero. `between_pair_direction_cosine` is the same over normalized
    M, which asks how aligned the pair-mean directions are once noise is averaged
    out. All three are exact functions of the summed unit deltas, so the bootstrap
    over windows is a matmul against multinomial weights.
    """
    U = np.stack([E[b] - E[a] for a, b in pairs]).astype(np.float64)  # (P, W, d)
    U /= np.linalg.norm(U, axis=2, keepdims=True).clip(1e-12)
    P, W, d = U.shape
    disjoint = np.array([[not ({a, b} & {c, e}) for c, e in pairs] for a, b in pairs])
    # (P*d, W) so each bootstrap resample is one gemv against the multinomial weights.
    Uf = np.ascontiguousarray(U.transpose(0, 2, 1).reshape(P * d, W))

    def stats(counts):
        S = (Uf @ counts).reshape(P, d)
        n = counts.sum()
        within = (((S ** 2).sum(1) - n) / (n * n - n)).mean()
        M = S / n
        Md = M / np.linalg.norm(M, axis=1, keepdims=True).clip(1e-12)
        return (float(within), float(np.abs(M @ M.T)[disjoint].mean()),
                float(np.abs(Md @ Md.T)[disjoint].mean()))

    point = stats(np.ones(W))
    rng = np.random.default_rng(seed)
    boots = np.array([stats(rng.multinomial(W, np.full(W, 1.0 / W)).astype(np.float64))
                      for _ in range(n_boot)])
    ci = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"n_windows": int(W), "n_pairs": int(P), "dim": int(d),
            "within_pair_cosine": point[0], "within_ci": ci[:, 0].tolist(),
            "between_pair_cosine": point[1], "between_ci": ci[:, 1].tolist(),
            "between_pair_direction_cosine": point[2], "between_direction_ci": ci[:, 2].tolist()}


def kit_transfer(E, pairs, centroids, seed=0):
    """Per-pair accuracy of e_A + mean(e_B - e_A) landing nearest kit B's centroid.

    The mean delta is estimated on half the windows and applied on the other half.
    Nearest is by cosine similarity to the train-split kit centroids.
    """
    W = E.shape[1]
    perm = np.random.default_rng(seed).permutation(W)
    fit, use = perm[: W // 2], perm[W // 2:]
    C = centroids / np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-12)
    acc = {}
    for a, b in pairs:
        v = (E[b, fit] - E[a, fit]).mean(0)
        z = E[a, use] + v
        z = z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)
        acc[(a, b)] = float(((z @ C.T).argmax(1) == b).mean())
    return acc


def mapping_table(pred, kit_id, row_kits, classes):
    """Row-normalized (len(row_kits), len(classes)) table of predicted-class shares."""
    T = np.zeros((len(row_kits), len(classes)))
    for i, k in enumerate(row_kits):
        p = pred[kit_id == k]
        T[i] = [(p == c).sum() for c in classes]
        T[i] /= max(1, T[i].sum())
    return T


def predictor_swap(ds, model, dev, amp, args, train_ids, kit_ids):
    """Two paired readings of the counterfactual swap, on the same windows and masks.

    `target_fixed` holds the target at kit B's x_{t+1} and varies only s_t: does the
    matching kit's state beat a train kit's state? Both arms are scored against the
    same target, so it is valid for held-out B as well. `target_swap` is the reverse,
    holding s_t at kit B and varying the target; that one is only reported for train
    B, because a held-out target is out of distribution for the teacher and the two
    targets are then not exchangeable.
    """
    key = window_keys(ds)
    uniq, inv = np.unique(key, return_inverse=True)
    loc = np.full((len(uniq), max(kit_ids) + 1), -1, np.int64)
    loc[inv, ds.kit_id] = np.arange(len(ds))
    full = np.flatnonzero((loc[:, kit_ids] >= 0).all(1))
    rng = np.random.default_rng(args.seed)
    w = rng.choice(full, min(args.max_swaps, len(full)), replace=False)
    a_kit = rng.choice(train_ids, len(w))
    b_kit = np.array([rng.choice([k for k in kit_ids if k != a]) for a in a_kit])
    i_a, i_b = loc[w, a_kit], loc[w, b_kit]

    chunk = max(1, args.bs // args.K)
    dl = DataLoader(SwapItems(ds, i_b, i_a), batch_size=chunk, num_workers=args.workers)
    # Three errors per window, all under the same masks: (state, target) in
    # {(B,B), (A,B), (B,A)}. (B,B) is the clean transition on kit B.
    arms = (("sB_tB", "x_t_B", "x_t1_B"), ("sA_tB", "x_t_A", "x_t1_B"),
            ("sB_tA", "x_t_B", "x_t1_A"))
    err = {a: [] for a, _, _ in arms}
    cols = {k: [] for k in ("seq_idx", "kit_A", "kit_B")}
    with torch.inference_mode():
        for b in tqdm.tqdm(dl, mininterval=5):
            n = b["tidx"].numel()
            mask = torch.cat([masks_for(model, i, args.K, args.seed, False)
                              for i in b["tidx"].tolist()]).to(dev)
            act = [b[k].to(dev).repeat_interleave(args.K, 0) for k in ("a_t1", "cc_t1")]
            for name, xk, tk in arms:
                x, x_t1 = (b[k].to(dev).repeat_interleave(args.K, 0) for k in (xk, tk))
                with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    e = model.state_prediction_error(x, *act, x_t1, mask)
                err[name].append(e.view(n, args.K).mean(1).float().cpu().numpy())
            for k in cols:
                cols[k].append(b[k].numpy())
    err = {k: np.concatenate(v) for k, v in err.items()}
    cols = {k: np.concatenate(v) for k, v in cols.items()}

    b_train = np.isin(cols["kit_B"], train_ids)
    subsets = (("all", np.ones(len(b_train), bool)), ("B_train_kit", b_train),
               ("B_heldout_kit", ~b_train))
    out = {"target_fixed": {}, "target_swap": {}}
    for comp, better, worse, keep in (("target_fixed", "sB_tB", "sA_tB", subsets),
                                      ("target_swap", "sB_tB", "sB_tA", subsets[1:2])):
        win = (err[better] < err[worse]).astype(np.float64)
        for name, sel in keep:
            lo, hi = cluster_bootstrap(win[sel], cols["seq_idx"][sel], seed=args.seed)
            out[comp][name] = {"n": int(sel.sum()), "win_rate": float(win[sel].mean()),
                               "ci": [lo, hi], f"err_{better}": float(err[better][sel].mean()),
                               f"err_{worse}": float(err[worse][sel].mean())}
    return out, int(len(full))


def pca2(X):
    """First two principal components of `X` (n, d) and their explained-variance shares."""
    Z = X - X.mean(0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :2] * S[:2], (S[:2] ** 2 / (S ** 2).sum()).tolist()


def confusion_figure(path, C, kits):
    """Row-normalized 14x14 confusion matrix of the 14-way linear teacher probe."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(kits)), kits, rotation=90, fontsize=6)
    ax.set_yticks(range(len(kits)), kits, fontsize=6)
    ax.set_xlabel("predicted kit")
    ax.set_ylabel("true kit")
    ax.set_title("E3: 14-way linear probe, teacher embedding (validation)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of clips")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def centroid_figure(path, panels, kits, n_train):
    """Three PCA panels of the 14 kit centroids: teacher, random-init encoder, raw mel."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (name, coords, var) in zip(axes, panels):
        for i, k in enumerate(kits):
            train = i < n_train
            ax.scatter(coords[i, 0], coords[i, 1], s=42, marker="o",
                       facecolors="#3b6ea5" if train else "none", edgecolors="#3b6ea5", lw=1.2)
            ax.annotate(k, coords[i], fontsize=6, xytext=(4, 3), textcoords="offset points")
        ax.set_title(f"{name} ({100 * sum(var):.0f}% var)", fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("E3: kit centroids, validation (filled = train kit, hollow = held out)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def verdict(res):
    """Six sentences on Q2, read off the metrics."""
    p = res["probes"]
    cons, t = res["geometry"]["consistency"], res["geometry"]["transfer"]
    g = cons["teacher"]
    s = res["geometry"]["predictor_swap"]
    held = res["heldout"]["expectations"]
    lin = {r: p["14way"][r]["linear"]["test"] for r in REPS}
    beats = [n for r, n in (("random", "the random-init encoder"), ("raw_mel", "the raw log-mel"))
             if lin["teacher"] > lin[r]]
    missed = [k for k, v in held.items() if not v]
    heldout_recall = res["per_kit_recall"]["test"][len(res["train_kits"]):]
    # Specificity of the displacement: how much more a delta agrees with another window
    # of its own kit pair than with a delta of an unrelated pair.
    ratio = {r: cons[r]["within_pair_cosine"] / cons[r]["between_pair_cosine"] for r in GEOM_REPS}
    ctrl = [r for r in GEOM_REPS if r != "teacher"]
    hi_within = [r for r in ctrl if cons[r]["within_pair_cosine"] > g["within_pair_cosine"]]
    hi_ratio = ratio["teacher"] > max(ratio[r] for r in ctrl)
    hi_transfer = t["teacher"]["all"] > max(t[r]["all"] for r in ctrl)
    if hi_ratio and hi_transfer:
        geom_note = ("both below the teacher, so this part of the geometry is built by "
                     "training and not inherited from the spectrum")
    elif not hi_ratio and not hi_transfer:
        geom_note = ("a control matching or beating the teacher on both, so the geometry is "
                     "inherited from the spectrum rather than built by training")
    else:
        geom_note = (f"the teacher ahead on {'the ratio' if hi_ratio else 'transfer'} but not "
                     f"on {'transfer' if hi_ratio else 'the ratio'}, so the geometry is only "
                     f"partly attributable to training")
    tf = res["geometry"]["predictor_swap"]["target_fixed"]
    won = [n for n in ("B_train_kit", "B_heldout_kit") if tf[n]["ci"][0] > 0.5]
    swap_note = (", so the predictor uses the kit of s_t, and it does so for unseen kits too"
                 if len(won) == 2 else
                 (", so the predictor uses the kit of s_t on train kits, but on unseen kits the "
                  "CI covers 0.5" if won else
                  ", so this does not show the predictor using the kit of s_t at all"))
    return [
        f"Kit identity is readable from s_t: the 14-way linear probe on the teacher "
        f"embedding scores {lin['teacher']:.3f} on test against {1 / 14:.3f} chance, and the "
        f"6-way probe on the train kits scores "
        f"{p['6way']['teacher']['linear']['test']:.3f} against {1 / 6:.3f}.",
        f"But notes/decisions.md sets the bar at beating both baselines, and on the 14-way task "
        f"the teacher beats "
        f"{'both of them' if len(beats) == 2 else (beats[0] if beats else 'neither')}: "
        f"random-init encoder {lin['random']:.3f}, raw mean log-mel {lin['raw_mel']:.3f}. "
        f"Kit identity is present in the embedding but "
        f"{'more' if lin['teacher'] > lin['raw_mel'] else 'no more'} linearly accessible there "
        f"than in the mel spectrum it was computed from, so the probe alone does not show the "
        f"model built a kit latent.",
        f"The counterfactual displacement e_B - e_A points the same way across performances "
        f"(teacher: within-pair cosine {g['within_pair_cosine']:.3f} "
        f"[{g['within_ci'][0]:.3f}, {g['within_ci'][1]:.3f}] against "
        f"{g['between_pair_cosine']:.3f} between disjoint kit pairs, a ratio of "
        f"{ratio['teacher']:.1f}) and a kit vector estimated on half the windows moves a clip "
        f"to the right kit's centroid on the other half {t['teacher']['all']:.3f} of the time "
        f"({t['teacher']['both_train']:.3f} between train kits, "
        f"{t['teacher']['both_heldout']:.3f} between two held-out kits).",
        "Run through the same pipeline the controls give within/between ratios of "
        + ", ".join(f"{ratio[r]:.1f} ({r})" for r in ctrl)
        + " against the teacher's " + f"{ratio['teacher']:.1f}, and transfer "
        + ", ".join(f"{t[r]['all']:.3f} ({r})" for r in ctrl) + " against "
        + f"{t['teacher']['all']:.3f} — {geom_note}"
        + ("" if not hi_within else
           "; " + " and ".join(hi_within)
           + (" has" if len(hi_within) == 1 else " have") + " a higher raw within-pair cosine "
           "than the teacher but spread it over unrelated kit pairs too, which is what the "
           "ratio and the transfer test catch") + ".",
        f"With the target held fixed at kit B's x_(t+1) and only s_t varying, giving the "
        f"predictor the matching kit's state beats giving it a train kit's state on "
        f"{s['target_fixed']['B_train_kit']['win_rate']:.3f} of windows "
        f"[{s['target_fixed']['B_train_kit']['ci'][0]:.3f}, "
        f"{s['target_fixed']['B_train_kit']['ci'][1]:.3f}] when B is a train kit and "
        f"{s['target_fixed']['B_heldout_kit']['win_rate']:.3f} "
        f"[{s['target_fixed']['B_heldout_kit']['ci'][0]:.3f}, "
        f"{s['target_fixed']['B_heldout_kit']['ci'][1]:.3f}] when B was never trained on"
        f"{swap_note}.",
        f"Held-out kits embed sensibly on the whole: {sum(held.values())} of {len(held)} send "
        f"most of their clips to the train kit the annotations predict"
        f"{'' if not missed else ' (' + ', '.join(missed) + ' did not)'}, and the lowest "
        f"14-way test recall among them is {min(heldout_recall):.3f} against {1 / 14:.3f} "
        f"chance, so the encoder places unseen kits in distinct, mostly plausible regions.",
    ]


def write_outputs(out_dir, res):
    """results.json and results.md; returns the markdown."""
    kits, n_train = res["kits"], len(res["train_kits"])
    md = [f"# E3 kit latent — {res['run_name']} (epoch {res['epoch']}), "
          f"{res['n_train_clips']} train clips, validation and test in full\n",
          "Clip embedding = mean over the 128 state tokens of the encoder output. The 14-way "
          "task is legitimate because the *training* clips of the held-out kits are embedded "
          "too: no kit was ever trained on by the JEPA, only by the probe. Chance is "
          f"{1 / 6:.3f} (6-way) and {1 / 14:.3f} (14-way).\n",
          "| task | representation | linear val | linear test | MLP val | MLP test |",
          "|---|---|---|---|---|---|"]
    for task in ("6way", "14way"):
        for rep in REPS:
            r = res["probes"][task][rep]
            md.append(f"| {task} | {rep} | {r['linear']['validation']:.3f} | "
                      f"{r['linear']['test']:.3f} | {r['mlp']['validation']:.3f} | "
                      f"{r['mlp']['test']:.3f} |")
    md += ["", "Per-kit recall, 14-way linear probe on the teacher embedding:", "",
           "| kit | split | validation | test |", "|---|---|---|---|"]
    for i, k in enumerate(kits):
        r = res["per_kit_recall"]
        md.append(f"| {k} | {'train' if i < n_train else 'held out'} | "
                  f"{r['validation'][i]:.3f} | {r['test'][i]:.3f} |")

    cons, t = res["geometry"]["consistency"], res["geometry"]["transfer"]
    g = cons["teacher"]
    md += ["", "## Counterfactual geometry (validation)", "",
           f"Ordered kit pairs: {g['n_pairs']}. Delta consistency on the same "
           f"{g['n_windows']} windows for every representation, 95% CI from a {N_BOOT}-resample "
           "bootstrap over windows. The random-init encoder and the raw mean log-mel are run "
           "through the identical pipeline, so the trained encoder's numbers only count if "
           "they beat both.", "",
           "| representation | within pair, different windows | between disjoint pairs, "
           "different windows | between disjoint pairs, pair-mean directions |",
           "|---|---|---|---|"]
    for rep in GEOM_REPS:
        c = cons[rep]
        md.append(f"| {rep} | {c['within_pair_cosine']:.3f} [{c['within_ci'][0]:.3f}, "
                  f"{c['within_ci'][1]:.3f}] | {c['between_pair_cosine']:.3f} "
                  f"[{c['between_ci'][0]:.3f}, {c['between_ci'][1]:.3f}] | "
                  f"{c['between_pair_direction_cosine']:.3f} "
                  f"[{c['between_direction_ci'][0]:.3f}, {c['between_direction_ci'][1]:.3f}] |")
    md += ["", "The first two columns are like for like: both average a cosine between two "
           "individual window deltas, so both carry the same per-window noise, and their ratio "
           "is what says whether a displacement is specific to its kit pair. Both between-pair "
           "columns are absolute cosines, because (D,C) is in the pair list whenever (C,D) is "
           "and carries the negated delta, so the signed mean is identically zero. The third "
           "column averages the noise out first and asks how aligned whole kit vectors are; "
           "two unrelated directions give about "
           + ", ".join(f"{np.sqrt(2 / (np.pi * d)):.3f} in {d}-d"
                       for d in sorted({cons[r]["dim"] for r in GEOM_REPS})) + ".", "",
           "Kit-vector transfer: the mean delta of a pair is estimated on half the windows, "
           "added to e_A on the other half, and scored by whether the nearest kit centroid "
           "(cosine, centroids from the train split of the same representation) is B.", "",
           "| representation | " + " | ".join(
               f"{n} (n={t['teacher'][f'{n}_n_pairs']})"
               for n in ("all", "both_train", "one_heldout", "both_heldout")) + " |",
           "|---|---|---|---|---|"]
    for rep in GEOM_REPS:
        md.append(f"| {rep} | " + " | ".join(
            f"{t[rep][n]:.3f}" for n in ("all", "both_train", "one_heldout", "both_heldout"))
            + " |")

    s = res["geometry"]["predictor_swap"]
    md += ["", "Predictor-side swap, target fixed. The target is kit B's x_{t+1} in both arms "
           "and only s_t changes, so the two errors are scored against the same teacher target "
           "and the comparison stays valid when B is a kit no encoder was trained on. Win rate "
           "is P[err with s_t from B < err with s_t from A]; A is always a train kit, B is "
           f"drawn from the other 13, same K=4 E1 masks. Validation has "
           f"{res['n_swap_windows']} windows present on all 14 kits and --max-swaps was "
           f"{res['max_swaps']}. CI: cluster bootstrap over sequences.", "",
           "| B | n | win rate [95% CI] | err s_t from B | err s_t from A |",
           "|---|---|---|---|---|"]
    for name, r in s["target_fixed"].items():
        md.append(f"| {name} | {r['n']} | {r['win_rate']:.3f} [{r['ci'][0]:.3f}, "
                  f"{r['ci'][1]:.3f}] | {r['err_sB_tB']:.4f} | {r['err_sA_tB']:.4f} |")
    md += ["", "The reverse reading, holding s_t at kit B and swapping the target, is reported "
           "for train-kit B only. With a held-out B the two targets are not exchangeable: the "
           "teacher never saw that kit, so the error against its target is inflated whatever "
           "the prediction, and the comparison measures target familiarity rather than kit "
           "tracking.", "",
           "| B | n | win rate [95% CI] | err vs target B | err vs target A |",
           "|---|---|---|---|---|"]
    for name, r in s["target_swap"].items():
        md.append(f"| {name} | {r['n']} | {r['win_rate']:.3f} [{r['ci'][0]:.3f}, "
                  f"{r['ci'][1]:.3f}] | {r['err_sB_tB']:.4f} | {r['err_sB_tA']:.4f} |")

    md += ["", "## Held-out kits", "",
           "Where the 6-way linear probe (teacher, trained on the 6 train kits only) sends "
           "each held-out kit's validation clips. Rows sum to 1; the expected column from "
           "configs/kits_v1.yaml is marked *.", "",
           "| held-out kit | " + " | ".join(res["train_kits"]) + " |",
           "|---" * (len(res["train_kits"]) + 1) + "|"]
    for i, k in enumerate(res["heldout_kits"]):
        cells = [f"{v:.2f}{'*' if res['train_kits'][j] in EXPECT[k] else ''}"
                 for j, v in enumerate(res["heldout"]["mapping"][i])]
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md += ["", "Nearest other kit by cosine distance between validation centroids:", "",
           "| kit | nearest | cosine distance |", "|---|---|---|"]
    for k, r in res["heldout"]["nearest_centroid"].items():
        md.append(f"| {k} | {r['kit']} | {r['distance']:.4f} |")
    md += ["", "## Q2", ""] + res["verdict"] + [""]

    md = "\n".join(md)
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    open(os.path.join(out_dir, "results.md"), "w").write(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/drumjepa_v1")
    ap.add_argument("--max-train", type=int, default=40000, help="train clips, 0 = all")
    ap.add_argument("--max-swaps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <run-dir>/e3")
    ap.add_argument("--recompute", action="store_true", help="ignore the embedding cache")
    args = ap.parse_args()
    args.K = 4  # same as E1/E2

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    dcfg = cfg["config"]["data"]
    dev = pick_device(args.device)
    amp = bool(cfg["config"]["amp"]) and dev.type != "cpu"
    out_dir = args.out or os.path.join(args.run_dir, "e3")
    os.makedirs(out_dir, exist_ok=True)

    ck = torch.load(os.path.join(args.run_dir, "last.pt"), map_location="cpu", weights_only=False)
    model = DrumJEPA(cfg["config"]["model"])
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval()
    torch.manual_seed(0)
    rand_model = DrumJEPA({}).to(dev).eval()
    models = {"teacher": (model, True), "student": (model, False), "random": (rand_model, True)}

    # kits=None: all 14 kits, so held-out kits are embedded too. Normalization from the
    # run config, never from the cache's stats.json (notes/decisions.md).
    dss = {s: SegmentPairs(dcfg["cache_dir"], s, seg_hop=dcfg["seg_hop"],
                           mel_mean=cfg["mel_mean"], mel_std=cfg["mel_std"]) for s in SPLITS}
    kits = dss["train"].meta["kits"]
    for key in ("MAP_VERSION", "split_version"):
        assert dss["train"].meta[key] == dcfg[key], key
    train_ids = [kits.index(k) for k in dcfg["train_kits"]]
    assert train_ids == list(range(len(train_ids))), "train kits must come first in meta['kits']"
    heldout_ids = [i for i in range(len(kits)) if i not in train_ids]
    all_ids = list(range(len(kits)))

    emb = {}
    for s in SPLITS:
        path = os.path.join(out_dir, f"emb_{s}.npz")
        if os.path.exists(path) and not args.recompute:
            emb[s] = dict(np.load(path))
            print(f"{s}: loaded {len(emb[s]['kit_id'])} cached embeddings")
            continue
        tidx = train_subset(dss[s], args.max_train, args.seed) if s == "train" \
            else np.arange(len(dss[s]))
        print(f"{s}: encoding {len(tidx)} clips on {dev} (bf16={amp})")
        emb[s] = encode_split(dss[s], tidx, models, dev, amp, args.bs, args.workers)
        np.savez(path, **emb[s])

    res = {"run_dir": args.run_dir, "run_name": cfg["config"]["run"]["name"],
           "epoch": ck["epoch"], "step": ck["step"], "seed": args.seed,
           "kits": kits, "train_kits": dcfg["train_kits"],
           "heldout_kits": [kits[i] for i in heldout_ids],
           "n_train_clips": int(len(emb["train"]["kit_id"])),
           "n_clips": {s: int(len(emb[s]["kit_id"])) for s in SPLITS},
           "chance": {"6way": 1 / 6, "14way": 1 / 14}, "n_bootstrap": N_BOOT,
           "git_commit": git_commit(), "probes": {}}

    print("probes: 6-way (train kits) then 14-way (all kits), 4 representations x 2 probes")
    res["probes"]["6way"], fitted6 = probe_scores(emb, train_ids, args.seed)
    res["probes"]["14way"], fitted14 = probe_scores(emb, all_ids, args.seed)

    scaler, clf = fitted14[("teacher", "linear")]
    res["per_kit_recall"], conf = {}, None
    for s in ("validation", "test"):
        pred = clf.predict(scaler.transform(emb[s]["teacher"]))
        y = emb[s]["kit_id"]
        res["per_kit_recall"][s] = [float((pred[y == k] == k).mean()) for k in all_ids]
        if s == "validation":
            conf = np.array([[(pred[y == k] == j).mean() for j in all_ids] for k in all_ids])
    confusion_figure(os.path.join(out_dir, "confusion.png"), conf, kits)
    res["confusion_validation"] = conf.tolist()

    print("geometry: deltas, kit-vector transfer, predictor-side swap")
    pairs = [(a, b) for a in all_ids for b in all_ids if a != b]
    groups = {"all": pairs,
              "both_train": [p for p in pairs if set(p) <= set(train_ids)],
              "one_heldout": [p for p in pairs if len(set(p) & set(heldout_ids)) == 1],
              "both_heldout": [p for p in pairs if set(p) <= set(heldout_ids)]}
    cons, transfer, sub = {}, {}, None
    for rep in GEOM_REPS:
        E = aligned_embeddings(emb["validation"], all_ids, rep)
        if sub is None:  # the same windows for every representation
            sub = np.random.default_rng(args.seed).choice(
                E.shape[1], min(N_GEOM_WINDOWS, E.shape[1]), replace=False)
        cons[rep] = delta_geometry(E[:, sub], pairs, N_BOOT, args.seed)
        acc = kit_transfer(E, pairs, kit_centroids(emb["train"], rep, all_ids), args.seed)
        transfer[rep] = {n: float(np.mean([acc[p] for p in ps])) for n, ps in groups.items()}
        transfer[rep].update({f"{n}_n_pairs": len(ps) for n, ps in groups.items()})
        transfer[rep]["per_pair"] = {f"{kits[a]} -> {kits[b]}": v for (a, b), v in acc.items()}
    swap, res["n_swap_windows"] = predictor_swap(dss["validation"], model, dev, amp, args,
                                                 train_ids, all_ids)
    res["max_swaps"] = args.max_swaps
    res["geometry"] = {"consistency": cons, "transfer": transfer, "predictor_swap": swap}

    scaler6, clf6 = fitted6[("teacher", "linear")]
    pred6 = clf6.predict(scaler6.transform(emb["validation"]["teacher"]))
    T = mapping_table(pred6, emb["validation"]["kit_id"], heldout_ids, train_ids)
    exp = {kits[k]: bool(dcfg["train_kits"][int(T[i].argmax())] in EXPECT[kits[k]])
           for i, k in enumerate(heldout_ids)}
    Cv = kit_centroids(emb["validation"], "teacher", all_ids)
    Cn = Cv / np.linalg.norm(Cv, axis=1, keepdims=True)
    D = 1.0 - Cn @ Cn.T
    np.fill_diagonal(D, np.inf)
    res["heldout"] = {"mapping": T.tolist(), "expectations": exp,
                      "centroid_cosine_distance": np.where(np.isinf(D), 0.0, D).tolist(),
                      "nearest_centroid": {kits[i]: {"kit": kits[int(D[i].argmin())],
                                                     "distance": float(D[i].min())}
                                           for i in all_ids}}

    panels = []
    for name, rep in (("teacher", "teacher"), ("random-init encoder", "random"),
                      ("raw log-mel", "raw_mel")):
        coords, var = pca2(kit_centroids(emb["validation"], rep, all_ids))
        panels.append((name, coords, var))
        res.setdefault("centroid_pca", {})[rep] = {"coords": coords.tolist(),
                                                   "explained_variance": var}
    centroid_figure(os.path.join(out_dir, "centroids.png"), panels, kits, len(train_ids))

    res["verdict"] = verdict(res)
    print(write_outputs(out_dir, res))


if __name__ == "__main__":
    main()
