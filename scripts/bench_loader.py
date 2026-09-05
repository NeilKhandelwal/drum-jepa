"""Benchmark the SegmentPairs dataloader before writing the model.

    python scripts/bench_loader.py [--cache data/cache/v1] [--split validation] [--workers 4]
"""
import argparse
import os
import sys
import time

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drumjepa.dataset import SegmentPairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/v1")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batches", type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()
    ds = SegmentPairs(args.cache, args.split)
    print(f"{args.split}: {len(ds):,} pairs from {ds.meta['n_files']} files / {ds.meta['n_sequences']} seqs, "
          f"index built in {time.time()-t0:.1f}s")
    print("density bins:", np.bincount(ds.density, minlength=7).tolist())
    print("pairs per kit:", np.bincount(ds.kit_id, minlength=len(ds.kits)).tolist())
    b = ds[0]
    print({k: tuple(v.shape) if hasattr(v, "shape") else v for k, v in b.items()})

    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    it = iter(dl); next(it)  # warm up
    t0 = time.time(); n = 0
    for _ in range(args.batches - 1):
        batch = next(it); n += 1
    dt = time.time() - t0
    mb = n * args.bs * (2 * 200 * 229 * 4 + 2 * 200 * 15 * 4) / 1e6
    print(f"{n/dt:.1f} batches/s, {n*args.bs/dt:,.0f} pairs/s, {mb/dt:.0f} MB/s of float32 tensors "
          f"(bs={args.bs}, workers={args.workers})")
    print("x_t batch stats: mean %.2f std %.2f" % (batch["x_t"].mean(), batch["x_t"].std()))


if __name__ == "__main__":
    main()
