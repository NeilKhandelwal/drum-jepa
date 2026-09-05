"""E2 state ablation: what does s_t carry when the action nearly determines the audio?

    python scripts/eval_e2.py --full runs/drumjepa_v1 --action-only runs/drumjepa_v1_actiononly

Both models see the same held-out transitions and the SAME E1 masks, so the errors
are paired. Q1 (CLAUDE.md) predicts a near tie on ordinary inputs and a win for the
full model exactly where s_t is the only source of information: post-ring-out
windows (few onsets in a_{t+1}, so the audio is the decay of what was ringing in
s_t) and under a full mask (no visible s_{t+1} tokens, so kit identity can only
come from s_t). Kit-swapped s_t moves the full model only; the action-only model is
invariant by construction (notes/decisions.md, "E2 action-only baseline"), which is
also the wiring check.

Writes <out>/{results.json,results.md,e2.png}, default <full>/e2.
"""
import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import torch
import tqdm
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drumjepa.dataset import SegmentPairs, density_bin  # noqa: E402
from drumjepa.model import DrumJEPA  # noqa: E402
from eval_e1 import build_sources, cluster_bootstrap, masks_for  # noqa: E402
from train import git_commit, pick_device  # noqa: E402

CONDS = ("clean", "kit_swap", "full_mask")
MODELS = ("full", "action_only")
# density_bin() bins on DENSITY_EDGES; 0-1 is <= 2 onsets, 5-6 is >= 21.
RING_OUT_BINS, DENSE_BINS = (0, 1), (5, 6)


def onset_bins(a_t1):
    """Onset count and density bin of each t+1 window, from a_t1 (B, T, K) -> two (B,) arrays.

    The dataset's `density` array is the *t* window, so the t+1 window is counted here.
    """
    n = (a_t1 > 0).flatten(1).sum(1).cpu().numpy().astype(np.int64)
    return n, np.array([density_bin(c) for c in n], np.int64)


def subset_masks(bins, n_onsets):
    """Boolean subset selectors keyed by name; `ring_out_le5` only if ring_out is thin."""
    sub = {"all": np.ones(len(bins), bool),
           "ring_out": np.isin(bins, RING_OUT_BINS),
           "dense": np.isin(bins, DENSE_BINS)}
    if sub["ring_out"].sum() < 100:
        sub["ring_out_le5"] = n_onsets <= 5
    return sub


class E2Items(Dataset):
    """One transition: its clean tensors plus the kit-swapped x_t."""

    def __init__(self, ds, tidx, src):
        self.ds, self.tidx, self.src = ds, tidx, src

    def __len__(self):
        return len(self.tidx)

    def __getitem__(self, j):
        i = int(self.tidx[j])
        it = self.ds[i]
        s = int(self.src["kit_swap"][i])
        return {"x_t": it["x_t"], "x_t1": it["x_t1"], "a_t1": it["a_t1"], "cc_t1": it["cc_t1"],
                "x_kit_swap": self.ds[s]["x_t"] if s >= 0 else it["x_t"],
                "ok_kit_swap": int(s >= 0), "tidx": i,
                "kit_id": it["kit_id"], "seq_idx": it["seq_idx"]}


def load_model(run_dir, dev):
    """(model in eval mode on `dev`, checkpoint, run config) for one run directory."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    ck = torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu", weights_only=False)
    model = DrumJEPA(cfg["config"]["model"])
    model.load_state_dict(ck["model"])
    return model.to(dev).eval(), ck, cfg


def check_comparable(full_cfg, ao_cfg):
    """Die loudly unless the two runs share data, normalization and versions."""
    for key in ("train_kits", "seg_hop", "MAP_VERSION", "split_version"):
        a, b = full_cfg["config"]["data"][key], ao_cfg["config"]["data"][key]
        assert a == b, f"data.{key}: full {a!r} != action-only {b!r}"
    for key in ("mel_mean", "mel_std"):
        assert full_cfg[key] == ao_cfg[key], f"{key}: full {full_cfg[key]} != action-only {ao_cfg[key]}"
    assert full_cfg["config"]["model"].get("use_state", True), "--full run has use_state false"
    assert not ao_cfg["config"]["model"].get("use_state", True), "--action-only run has use_state true"


def aggregate(e_full, e_ao, seq, seed):
    """Paired stats for one (condition, subset): n, means, win rate, log ratio, CIs."""
    assert (e_full > 0).all() and (e_ao > 0).all(), "zero prediction error: log ratio undefined"
    win = (e_full < e_ao).astype(np.float64)
    lr = np.log(e_ao / e_full)
    wlo, whi = cluster_bootstrap(win, seq, seed=seed)
    llo, lhi = cluster_bootstrap(lr, seq, seed=seed)
    return {"n": int(len(win)), "err_full": float(e_full.mean()), "err_action_only": float(e_ao.mean()),
            "win_rate": float(win.mean()), "win_ci": [wlo, whi],
            "log_ratio": float(lr.mean()), "log_ratio_ci": [llo, lhi]}


def verdict(m):
    """Three lines on whether Q1's prediction held, read off the metrics."""
    def d(c, s):
        r = m[c][s]
        lo, hi = r["log_ratio_ci"]
        sign = "full better" if lo > 0 else ("action-only better" if hi < 0 else "tie (CI spans 0)")
        return (f"n={r['n']}, win {r['win_rate']:.3f}, log-ratio {r['log_ratio']:+.4f} "
                f"[{lo:+.4f}, {hi:+.4f}] -> {sign}")
    ring = next(s for s in ("ring_out", "ring_out_le5", "all") if s in m["clean"])
    return [f"1. Clean, all: {d('clean', 'all')}. Q1 predicts a near tie.",
            f"2. Clean, {ring}: {d('clean', ring)}. Q1 predicts the full model wins.",
            f"3. Full mask, all: {d('full_mask', 'all')}; kit-swapped s_t, all: "
            f"{d('kit_swap', 'all')}. Q1 predicts the full model wins under a full mask."]


def write_outputs(out_dir, res):
    """results.json, results.md and e2.png; returns the markdown."""
    rows = ["| condition | subset | n | err full | err action-only | win rate [95% CI] "
            "| log-ratio [95% CI] |", "|---|---|---|---|---|---|---|"]
    for c, subs in res["metrics"].items():
        for s, r in subs.items():
            wc, lc = r["win_ci"], r["log_ratio_ci"]
            rows.append(f"| {c} | {s} | {r['n']} | {r['err_full']:.4f} | "
                        f"{r['err_action_only']:.4f} | "
                        f"{r['win_rate']:.3f} [{wc[0]:.3f}, {wc[1]:.3f}] | "
                        f"{r['log_ratio']:+.4f} [{lc[0]:+.4f}, {lc[1]:+.4f}] |")
    ks = res["kit_swap_self"]
    md = (f"# E2 state ablation — full (epoch {res['epoch_full']}) vs action-only "
          f"(epoch {res['epoch_action_only']}), {res['split']}\n\n"
          f"{res['n_transitions']} transitions, K={res['K']} (full_mask: K=1), seed {res['seed']}. "
          f"Win rate is P[err full < err action-only]; log-ratio is "
          f"mean log(err action-only / err full), positive = full better. "
          f"CIs: 1000-resample cluster bootstrap over sequences.\n\n"
          + "\n".join(rows) + "\n\n"
          f"Own clean-vs-kit-swapped win rate (the E1 number, n={ks['n']}): "
          f"full {ks['full']:.3f}, action-only {ks['action_only']:.3f} "
          f"(invariant by construction, so every pair ties and ties count as half; "
          f"max |clean - swapped| per transition {ks['action_only_max_abs_diff']:.2e}).\n\n"
          + (f"{res['ring_out_note']}\n\n" if res.get("ring_out_note") else "")
          + "## Q1\n\n" + "\n".join(res["verdict"]) + "\n")
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    open(os.path.join(out_dir, "results.md"), "w").write(md)

    pairs = [(c, s) for c, subs in res["metrics"].items() for s in subs]
    y = np.arange(len(pairs))[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, colour, off in (("full", "#3b6ea5", 0.19), ("action_only", "#a37b3b", -0.19)):
        ax.barh(y + off, [res["metrics"][c][s][f"err_{name}"] for c, s in pairs],
                height=0.36, color=colour, label=name)
    hi = [max(res["metrics"][c][s]["err_full"], res["metrics"][c][s]["err_action_only"])
          for c, s in pairs]
    xmax = max(hi)
    for i, (c, s) in enumerate(pairs):
        r = res["metrics"][c][s]
        ax.text(hi[i] + 0.02 * xmax, y[i],
                f"{r['win_rate']:.2f}", va="center", fontsize=7)
    ax.set_yticks(y, [f"{c} / {s}" for c, s in pairs], fontsize=7)
    ax.set_xlim(0, xmax * 1.18)
    ax.set_xlabel("mean state prediction error (win rate at right)")
    ax.set_title(f"E2: state ablation, {res['split']}")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e2.png"), dpi=150)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="runs/drumjepa_v1")
    ap.add_argument("--action-only", default="runs/drumjepa_v1_actiononly")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all transitions")
    ap.add_argument("--K", type=int, default=4, help="masks averaged per transition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=128, help="model batch = transitions x K")
    ap.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <full>/e2")
    args = ap.parse_args()

    dev = pick_device(args.device)
    models, cks, cfgs = {}, {}, {}
    for name, run_dir in (("full", args.full), ("action_only", args.action_only)):
        models[name], cks[name], cfgs[name] = load_model(run_dir, dev)
    check_comparable(cfgs["full"], cfgs["action_only"])
    cfg, dcfg = cfgs["full"], cfgs["full"]["config"]["data"]
    amp = bool(cfg["config"]["amp"]) and dev.type != "cpu"

    # Normalization from the full run's config, never from the cache's stats.json.
    ds = SegmentPairs(dcfg["cache_dir"], args.split, seg_hop=dcfg["seg_hop"],
                      kits=dcfg["train_kits"], mel_mean=cfg["mel_mean"], mel_std=cfg["mel_std"])
    for key in ("MAP_VERSION", "split_version"):
        assert ds.meta[key] == dcfg[key], f"{key}: cache {ds.meta[key]} != run {dcfg[key]}"

    rng = np.random.default_rng(args.seed)
    src = build_sources(ds, dcfg["seg_hop"], rng)
    tidx = np.arange(len(ds))
    if args.max_pairs:
        tidx = np.sort(rng.choice(len(ds), min(args.max_pairs, len(ds)), replace=False))
    chunk = max(1, args.bs // args.K)
    dl = DataLoader(E2Items(ds, tidx, src), batch_size=chunk, num_workers=args.workers)
    print(f"device={dev} amp(bf16)={amp} split={args.split} transitions={len(tidx)} "
          f"K={args.K} chunk={chunk}")

    errs = {(m, c): [] for m in MODELS for c in CONDS}
    cols = {k: [] for k in ("seq_idx", "kit_id", "ok_kit_swap", "n_onsets", "bin")}
    with torch.inference_mode():
        for b in tqdm.tqdm(dl, mininterval=5):
            n = b["tidx"].numel()
            # Masks come from the full model; both models share n_s and the mask config,
            # and E1's seeding makes the numbers comparable to docs/e1/.
            mask = torch.cat([masks_for(models["full"], i, args.K, args.seed, False)
                              for i in b["tidx"].tolist()]).to(dev)
            full_mask = torch.ones(n, models["full"].n_s, dtype=torch.bool, device=dev)
            x_t1, a_t1, cc_t1 = (b[k].to(dev) for k in ("x_t1", "a_t1", "cc_t1"))
            rep = [t.repeat_interleave(args.K, 0) for t in (a_t1, cc_t1, x_t1)]
            for m, model in models.items():
                for c in CONDS:
                    x = b["x_kit_swap" if c == "kit_swap" else "x_t"].to(dev)
                    with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                        if c == "full_mask":
                            e = model.state_prediction_error(x, a_t1, cc_t1, x_t1, full_mask)
                        else:
                            e = model.state_prediction_error(x.repeat_interleave(args.K, 0), *rep, mask)
                            e = e.view(n, args.K).mean(1)
                    errs[(m, c)].append(e.float().cpu().numpy())
            n_on, bins = onset_bins(a_t1)
            cols["n_onsets"].append(n_on)
            cols["bin"].append(bins)
            for k in ("seq_idx", "kit_id", "ok_kit_swap"):
                cols[k].append(b[k].numpy())

    errs = {k: np.concatenate(v) for k, v in errs.items()}
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    ok = cols["ok_kit_swap"].astype(bool)
    subs = subset_masks(cols["bin"], cols["n_onsets"])

    diff = np.abs(errs[("action_only", "clean")] - errs[("action_only", "kit_swap")])[ok]
    assert diff.max() < 1e-3, f"action-only model is not kit-swap invariant: max diff {diff.max():.3e}"

    res = {"full": args.full, "action_only": args.action_only, "split": args.split,
           "epoch_full": cks["full"]["epoch"], "epoch_action_only": cks["action_only"]["epoch"],
           "step_full": cks["full"]["step"], "step_action_only": cks["action_only"]["step"],
           "n_transitions": int(len(tidx)), "K": args.K, "seed": args.seed, "n_bootstrap": 1000,
           "git_commit": git_commit(), "metrics": {}}
    for c in CONDS:
        res["metrics"][c] = {}
        for s, sel in subs.items():
            m = sel & ok if c == "kit_swap" else sel
            if not m.any():
                continue
            res["metrics"][c][s] = aggregate(errs[("full", c)][m], errs[("action_only", c)][m],
                                             cols["seq_idx"][m], args.seed)
    res["subset_n"] = {s: int(sel.sum()) for s, sel in subs.items()}
    if "ring_out_le5" in subs:
        res["ring_out_note"] = (f"ring_out (<= 2 onsets in the t+1 window) has only "
                                f"{res['subset_n']['ring_out']} transitions, so the <= 5 onset "
                                f"subset `ring_out_le5` ({res['subset_n']['ring_out_le5']}) is "
                                f"reported as well.")
    # Ties count as half here (unlike E1's strict <): the action-only model's two
    # errors are bitwise identical, and 0.5 is the right reading of "no preference".
    res["kit_swap_self"] = {
        "n": int(ok.sum()),
        "action_only_max_abs_diff": float(diff.max()),
        **{m: float(np.mean(np.where(errs[(m, "clean")][ok] == errs[(m, "kit_swap")][ok], 0.5,
                                     errs[(m, "clean")][ok] < errs[(m, "kit_swap")][ok])))
           for m in MODELS}}
    res["verdict"] = verdict(res["metrics"])

    out_dir = args.out or os.path.join(args.full, "e2")
    os.makedirs(out_dir, exist_ok=True)
    print(write_outputs(out_dir, res))


if __name__ == "__main__":
    main()
