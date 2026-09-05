"""Train drum-JEPA on the E-GMD segment-pair cache.

    python scripts/train.py --config configs/drumjepa_v1.yaml
    python scripts/train.py --config configs/drumjepa_v1.yaml --overfit 32 --epochs 200
    python scripts/train.py --config configs/drumjepa_v1.yaml --resume

Writes to the run directory: config.json (resolved config, normalization stats,
parameter counts, git commit), metrics.csv (one row per epoch) and last.pt
(model, optimizer, scheduler, epoch, step, RNG state) saved every epoch.

The overfit mode is the wiring test from CLAUDE.md step 5: on a few dozen fixed
segments the loss must go toward ~0. Watch the *_std columns in either mode --
a falling loss with shrinking embedding std is collapse, not learning.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import tqdm
import yaml
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drumjepa.dataset import SegmentPairs  # noqa: E402
from drumjepa.model import DrumJEPA  # noqa: E402

MON = ("loss", "loss_s", "loss_a", "s_tea_std", "s_stu_std", "a_tea_std", "a_stu_std")
FIELDS = (["epoch", "step"] + [f"train/{k}" for k in MON]
          + ["val/loss", "val/loss_s", "val/loss_a", "val/s_tea_std", "lr", "epoch_s"])


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_dev(batch, dev):
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/drumjepa_v1.yaml")
    ap.add_argument("--run-dir", default=None, help="default: runs/<run.name>")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke test)")
    ap.add_argument("--overfit", type=int, default=0, help="train on N fixed pairs, no validation")
    ap.add_argument("--resume", action="store_true", help="continue from <run-dir>/last.pt")
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50, help="steps between mid-epoch log lines")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dcfg, ocfg = cfg["data"], cfg["optim"]
    if args.epochs is not None:
        ocfg["epochs"] = args.epochs
    if args.workers is not None:
        dcfg["workers"] = args.workers
    dev = pick_device(args.device)
    amp = bool(cfg["amp"]) and not args.no_amp and dev.type != "cpu"
    run_dir = args.run_dir or os.path.join("runs", cfg["run"]["name"])
    os.makedirs(run_dir, exist_ok=True)
    seed = cfg["run"]["seed"]
    seed_all(seed)
    print(f"device={dev} amp(bf16)={amp} run_dir={run_dir} overfit={args.overfit}")

    train_ds = SegmentPairs(dcfg["cache_dir"], "train", seg_hop=dcfg["seg_hop"], kits=dcfg["train_kits"])
    for key in ("MAP_VERSION", "split_version"):  # fail loud on a cache/config mismatch
        assert train_ds.meta[key] == dcfg[key], f"{key}: cache {train_ds.meta[key]} != config {dcfg[key]}"
    val_dl = None
    if args.overfit:
        idx = np.random.default_rng(seed).choice(len(train_ds), args.overfit, replace=False)
        train_set, bs = Subset(train_ds, sorted(idx.tolist())), min(ocfg["batch_size"], args.overfit)
    else:
        train_set, bs = train_ds, ocfg["batch_size"]
        val_ds = SegmentPairs(dcfg["cache_dir"], "validation", seg_hop=dcfg["seg_hop"],
                              kits=dcfg["train_kits"], mel_mean=train_ds.mel_mean, mel_std=train_ds.mel_std)
        n_val = min(dcfg["val_pairs"], len(val_ds))
        vidx = np.random.default_rng(seed).choice(len(val_ds), n_val, replace=False)
        val_dl = DataLoader(Subset(val_ds, sorted(vidx.tolist())), batch_size=bs,
                            num_workers=dcfg["workers"], persistent_workers=dcfg["workers"] > 0)
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=not args.overfit,
                          num_workers=dcfg["workers"], persistent_workers=dcfg["workers"] > 0, generator=g)
    print(f"train pairs: {len(train_set):,} on {len(train_ds.kit_names_used)} kits, {len(train_dl)} steps/epoch"
          + ("" if val_dl is None else f"; val pairs: {len(val_dl.dataset):,}"))

    model = DrumJEPA(cfg["model"]).to(dev)
    n_params = model.n_params()
    print("params: " + " ".join(f"{k}={v/1e6:.1f}M" for k, v in n_params.items()))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=ocfg["lr"],
                            betas=tuple(ocfg["betas"]), weight_decay=ocfg["weight_decay"])
    warm = min(ocfg["warmup_steps"], max(1, len(train_dl) * ocfg["epochs"] - 1))
    total = max(warm + 1, len(train_dl) * ocfg["epochs"])
    sched = SequentialLR(opt, milestones=[warm], schedulers=[
        LinearLR(opt, start_factor=1e-2, total_iters=warm),
        CosineAnnealingLR(opt, T_max=total - warm, eta_min=ocfg["min_lr_factor"] * ocfg["lr"])])

    csv_path, ckpt_path = os.path.join(run_dir, "metrics.csv"), os.path.join(run_dir, "last.pt")
    start_epoch, step = 0, 0
    if args.resume:
        # map_location="cpu": the RNG states are ByteTensors and torch.set_rng_state
        # rejects them if torch.load has moved them to the accelerator.
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["random"])
        g.set_state(ck["rng"]["loader"])
        print(f"resumed from {ckpt_path}: epoch={start_epoch} step={step}")
    else:
        json.dump({"config": cfg, "args": vars(args), "MAP_VERSION": dcfg["MAP_VERSION"],
                   "split_version": dcfg["split_version"], "mel_mean": train_ds.mel_mean,
                   "mel_std": train_ds.mel_std, "n_params": n_params, "device": dev.type,
                   "amp_bf16": amp, "torch": torch.__version__, "git_commit": git_commit()},
                  open(os.path.join(run_dir, "config.json"), "w"), indent=1)
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    for epoch in range(start_epoch, ocfg["epochs"]):
        model.train()
        t0, sums, n = time.time(), {k: 0.0 for k in MON}, 0
        for batch in tqdm.tqdm(train_dl, desc=f"epoch {epoch}", leave=False, mininterval=5):
            batch = to_dev(batch, dev)
            # Forward and loss stay inside autocast: a bf16 activation reaching an fp32
            # Linear outside it aborts the MPS process (notes/mps_fixes.md).
            with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                out = model(batch)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            if ocfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg["grad_clip"])
            opt.step()
            sched.step()
            model.ema_update()  # fp32, outside autocast; once per optimizer step
            step += 1
            n += 1
            for k in MON:
                sums[k] += out[k].item()
            if args.log_every and step % args.log_every == 0:
                print(f"  step={step} " + " ".join(f"{k}={sums[k]/n:.4g}" for k in
                                                   ("loss", "loss_s", "loss_a", "s_tea_std", "a_tea_std"))
                      + f" lr={opt.param_groups[0]['lr']:.3g}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        row = {"epoch": epoch, "step": step, "lr": opt.param_groups[0]["lr"],
               "epoch_s": round(time.time() - t0, 1),
               **{f"train/{k}": sums[k] / max(n, 1) for k in MON},
               **{f"val/{k}": "" for k in ("loss", "loss_s", "loss_a", "s_tea_std")}}
        if val_dl is not None:
            model.eval()
            vs, vn = {k: 0.0 for k in ("loss", "loss_s", "loss_a", "s_tea_std")}, 0
            with torch.inference_mode():
                for batch in val_dl:
                    with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                        out = model(to_dev(batch, dev))
                    for k in vs:
                        vs[k] += out[k].item()
                    vn += 1
            row.update({f"val/{k}": v / max(vn, 1) for k, v in vs.items()})
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(row)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "step": step, "config": cfg, "args": vars(args),
                    "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                            "random": random.getstate(), "loader": g.get_state()}}, ckpt_path)
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
