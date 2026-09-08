"""E1 dynamics sanity check: does the predictor f(s_t, a_{t+1}) use its inputs?

    python scripts/eval_e1.py --run-dir runs/drumjepa_v1

For every held-out transition we score the state prediction error, then corrupt
exactly one input (the target x_{t+1} is always clean) and rescore with the SAME
masks. The win rate is the fraction of transitions where the clean error is lower.
Gate (CLAUDE.md, E1): win rates clearly above 0.5. A predictor that ignores s_t
scores ~0.5 on the state perturbations and exactly ~0.5 on kit swap.

Writes <run-dir>/e1/{results.json,results.md,win_rates.png}.
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
from drumjepa.dataset import SegmentPairs  # noqa: E402
from drumjepa.model import DrumJEPA  # noqa: E402
from train import git_commit, pick_device  # noqa: E402

PERTS = ("state_shift", "state_random", "action_shift", "action_random", "kit_swap")
CONDS = ("clean",) + PERTS
X_SUB = ("state_shift", "state_random", "kit_swap")  # perturbations that replace x_t


def random_other_seq(seq_idx, rng, tries=64):
    """For each transition, a uniform transition from a different sequence (-1 if none)."""
    n = len(seq_idx)
    out = rng.integers(0, n, n)
    for _ in range(tries):
        bad = seq_idx[out] == seq_idx
        if not bad.any():
            break
        out[bad] = rng.integers(0, n, int(bad.sum()))
    out[seq_idx[out] == seq_idx] = -1
    return out


def build_sources(ds, seg_hop, rng):
    """Source transition index per perturbation, (n,) int64 arrays, -1 = unavailable.

    Built from the dataset index arrays: (seq_idx, kit_id, offset) locates the
    time-shifted neighbour, (seq_idx, offset) the same window on the other kits.
    """
    by_key, by_win = {}, {}
    for i in range(len(ds)):
        key = (int(ds.seq_idx[i]), int(ds.offset[i]))
        by_key[(key[0], int(ds.kit_id[i]), key[1])] = i
        by_win.setdefault(key, []).append(i)
    src = {p: np.full(len(ds), -1, np.int64) for p in ("state_shift", "kit_swap")}
    for i in range(len(ds)):
        s, k, o = int(ds.seq_idx[i]), int(ds.kit_id[i]), int(ds.offset[i])
        # Earlier window only. The later window's x_t is this transition's own x_{t+1}
        # at non-overlapping hop: an easier input, not a corrupted one. Skip instead.
        if (s, k, o - seg_hop) in by_key:
            src["state_shift"][i] = by_key[(s, k, o - seg_hop)]
        others = [j for j in by_win[(s, o)] if int(ds.kit_id[j]) != k]
        if others:
            src["kit_swap"][i] = others[rng.integers(len(others))]
    src["state_random"] = random_other_seq(ds.seq_idx, rng)
    src["action_random"] = random_other_seq(ds.seq_idx, rng)
    src["action_shift"] = np.arange(len(ds), dtype=np.int64)  # (a_t, cc_t) of the item itself
    return src


class E1Items(Dataset):
    """One transition: its clean tensors plus the source tensors each perturbation swaps in."""

    def __init__(self, ds, tidx, src):
        self.ds, self.tidx, self.src = ds, tidx, src

    def __len__(self):
        return len(self.tidx)

    def __getitem__(self, j):
        i = int(self.tidx[j])
        it = self.ds[i]
        out = {k: it[k] for k in ("x_t", "x_t1", "a_t", "a_t1", "cc_t", "cc_t1")}
        out.update(tidx=i, kit_id=it["kit_id"], seq_idx=it["seq_idx"], ok_action_shift=1)
        for p in X_SUB:
            s = int(self.src[p][i])
            out[f"ok_{p}"] = int(s >= 0)
            out[f"x_{p}"] = self.ds[s]["x_t"] if s >= 0 else it["x_t"]
        s = int(self.src["action_random"][i])
        o = self.ds[s] if s >= 0 else it
        out.update(ok_action_random=int(s >= 0), a_action_random=o["a_t1"],
                   cc_action_random=o["cc_t1"])
        return out


def cond_inputs(b, cond):
    """(x_t, a_{t+1}, cc_{t+1}) for one condition; the target x_{t+1} is always clean."""
    if cond in X_SUB:
        return b[f"x_{cond}"], b["a_t1"], b["cc_t1"]
    if cond == "action_shift":
        return b["x_t"], b["a_t"], b["cc_t"]
    if cond == "action_random":
        return b["x_t"], b["a_action_random"], b["cc_action_random"]
    return b["x_t"], b["a_t1"], b["cc_t1"]


def masks_for(model, i, K, seed, full):
    """The K masks of transition `i`, (K, n_s) bool, deterministic in (seed, i).

    The same masks are reused for the clean and every perturbed condition, so the
    comparison is paired.
    """
    if full:
        return torch.ones(1, model.n_s, dtype=torch.bool)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((seed * 1_000_003 + i) % 2**31)
        return model._state_mask(K, torch.device("cpu"))


def cluster_bootstrap(win, seq, n_boot=1000, seed=0):
    """95% CI on a win rate, resampling SEQUENCES: kits share performances."""
    _, inv = np.unique(seq, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    rng = np.random.default_rng(seed)
    boots = [win[np.concatenate([groups[p] for p in rng.integers(0, len(groups), len(groups))])].mean()
             for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def write_outputs(out_dir, res, run_name):
    """results.json, results.md and win_rates.png; returns the markdown."""
    rows = ["| perturbation | n | win rate [95% CI] | clean err | perturbed err |",
            "|---|---|---|---|---|"]
    for p in PERTS:
        r = res["perturbations"][p]
        rows.append(f"| {p} | {r['n']} | {r['win_rate']:.3f} "
                    f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | "
                    f"{r['clean_err']:.4f} | {r['perturbed_err']:.4f} |")
    rows += ["", "kit_swap by the kit of the clean x_t:", "",
             "| kit | n | win rate |", "|---|---|---|"]
    rows += [f"| {k} | {v['n']} | {v['win_rate']:.3f} |" for k, v in res["kit_swap_by_kit"].items()]
    md = (f"# E1 dynamics sanity — {run_name} (epoch {res['epoch']}, {res['split']}, "
          f"{res['kits']} kits)\n\n"
          f"{res['n_transitions']} transitions, K={res['K']}"
          f"{' (full mask)' if res['full_mask'] else ''}, seed {res['seed']}.\n\n"
          + "\n".join(rows) + "\n")
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    open(os.path.join(out_dir, "results.md"), "w").write(md)

    fig, ax = plt.subplots(figsize=(6, 3))
    y = np.arange(len(PERTS))[::-1]
    wr = np.array([res["perturbations"][p]["win_rate"] for p in PERTS])
    ci = np.array([res["perturbations"][p]["ci"] for p in PERTS])
    ax.barh(y, wr, height=0.6, color="#3b6ea5",
            xerr=[wr - ci[:, 0], ci[:, 1] - wr], error_kw=dict(ecolor="#222", capsize=3, lw=1))
    ax.axvline(0.5, ls="--", c="#a33", lw=1)
    ax.set_yticks(y, list(PERTS))
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("win rate (clean error < perturbed error)")
    ax.set_title(f"E1: {run_name}, epoch {res['epoch']}")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "win_rates.png"), dpi=150)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/drumjepa_v1")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--kits", default="train", choices=["train", "heldout", "all"],
                    help="kit subset to evaluate on; heldout/all write to <run-dir>/e1_<kits>")
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all transitions")
    ap.add_argument("--K", type=int, default=4, help="masks averaged per transition")
    ap.add_argument("--full-mask", action="store_true", help="one all-masked mask instead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=128, help="model batch = transitions x K")
    ap.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    dcfg = cfg["config"]["data"]
    K = 1 if args.full_mask else args.K
    dev = pick_device(args.device)
    amp = bool(cfg["config"]["amp"]) and dev.type != "cpu"

    # Kit subset. The cache holds all 14 kits of configs/kits_v1.yaml; a run trains on 6.
    # Held-out kits are the cache's kits minus the run's train kits, as in eval_e3.py.
    # Every perturbation source, kit swap included, is drawn from this subset.
    all_kits = json.load(open(os.path.join(dcfg["cache_dir"], args.split, "meta.json")))["kits"]
    kit_subset = {"train": list(dcfg["train_kits"]),
                  "heldout": [k for k in all_kits if k not in dcfg["train_kits"]],
                  "all": list(all_kits)}[args.kits]

    # Normalization comes from the run config, never from the cache's stats.json
    # (notes/decisions.md: a rebuilt cache changes stats.json silently).
    ds = SegmentPairs(dcfg["cache_dir"], args.split, seg_hop=dcfg["seg_hop"],
                      kits=kit_subset, mel_mean=cfg["mel_mean"], mel_std=cfg["mel_std"])
    for key in ("MAP_VERSION", "split_version"):
        assert ds.meta[key] == dcfg[key], f"{key}: cache {ds.meta[key]} != run {dcfg[key]}"

    ck = torch.load(os.path.join(args.run_dir, "last.pt"), map_location="cpu", weights_only=False)
    model = DrumJEPA(cfg["config"]["model"])
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval()

    rng = np.random.default_rng(args.seed)
    src = build_sources(ds, dcfg["seg_hop"], rng)
    tidx = np.arange(len(ds))
    if args.max_pairs:
        tidx = np.sort(rng.choice(len(ds), min(args.max_pairs, len(ds)), replace=False))
    chunk = max(1, args.bs // K)
    dl = DataLoader(E1Items(ds, tidx, src), batch_size=chunk, num_workers=args.workers)
    print(f"device={dev} amp(bf16)={amp} split={args.split} transitions={len(tidx)} "
          f"K={K} full_mask={args.full_mask} chunk={chunk}")

    errs = {c: [] for c in CONDS}
    meta = {k: [] for k in ("kit_id", "seq_idx")}
    ok = {p: [] for p in PERTS}
    with torch.inference_mode():
        for b in tqdm.tqdm(dl, mininterval=5):
            mask = torch.cat([masks_for(model, i, K, args.seed, args.full_mask)
                              for i in b["tidx"].tolist()]).to(dev)
            n = b["tidx"].numel()
            x_t1 = b["x_t1"].to(dev).repeat_interleave(K, 0)
            for c in CONDS:
                ins = [t.to(dev).repeat_interleave(K, 0) for t in cond_inputs(b, c)]
                with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    e = model.state_prediction_error(*ins, x_t1, mask)
                errs[c].append(e.view(n, K).mean(1).float().cpu().numpy())
            for k in meta:
                meta[k].append(b[k].numpy())
            for p in PERTS:
                ok[p].append(b[f"ok_{p}"].numpy().astype(bool))

    errs = {c: np.concatenate(v) for c, v in errs.items()}
    kit_id, seq_idx = (np.concatenate(meta[k]) for k in ("kit_id", "seq_idx"))
    ok = {p: np.concatenate(v) for p, v in ok.items()}
    kits = ds.meta["kits"]

    res = {"run_dir": args.run_dir, "split": args.split, "epoch": ck["epoch"], "step": ck["step"],
           "kits": args.kits, "kit_names": kit_subset,
           "n_transitions": int(len(tidx)), "K": K, "full_mask": args.full_mask,
           "seed": args.seed, "n_bootstrap": 1000, "git_commit": git_commit(),
           "perturbations": {}, "skipped": {}, "kit_swap_by_kit": {}}
    for p in PERTS:
        m = ok[p]
        win = (errs["clean"][m] < errs[p][m]).astype(np.float64)
        lo, hi = cluster_bootstrap(win, seq_idx[m], seed=args.seed)
        res["perturbations"][p] = {"n": int(m.sum()), "win_rate": float(win.mean()),
                                   "ci": [lo, hi], "clean_err": float(errs["clean"][m].mean()),
                                   "perturbed_err": float(errs[p][m].mean())}
        res["skipped"][p] = int((~m).sum())
    m = ok["kit_swap"]
    for k in np.unique(kit_id[m]):
        sel = m & (kit_id == k)
        win = errs["clean"][sel] < errs["kit_swap"][sel]
        res["kit_swap_by_kit"][kits[int(k)]] = {"n": int(sel.sum()), "win_rate": float(win.mean())}

    out_dir = os.path.join(args.run_dir, "e1" if args.kits == "train" else f"e1_{args.kits}")
    os.makedirs(out_dir, exist_ok=True)
    print(write_outputs(out_dir, res, cfg["config"]["run"]["name"]))
    print("skipped (perturbation unavailable): " + json.dumps(res["skipped"]))


if __name__ == "__main__":
    main()
