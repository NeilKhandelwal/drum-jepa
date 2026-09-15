"""Write <cache_dir>/stats.json: the train split's log-mel mean and std over 1M random
frames, seed 0 (the recipe recorded in data/cache/v1/stats.json). SegmentPairs reads
this file for default normalization (drumjepa/dataset.py); runs record the values
they used in config.json (notes/decisions.md, "Mel normalization").

    .venv/bin/python scripts/cache_stats.py --cache-dir data/cache/v2
"""
import argparse
import json
import os

import numpy as np

from drumjepa.features import N_MELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--n-frames", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    d = os.path.join(args.cache_dir, "train")
    meta = json.load(open(os.path.join(d, "meta.json")))
    mel = np.memmap(os.path.join(d, "mel.f16"), dtype=np.float16, mode="r",
                    shape=(meta["mel_frames"], N_MELS))
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(meta["mel_frames"], size=args.n_frames, replace=False))
    x = mel[idx].astype(np.float32)
    stats = {"mel_mean": float(x.mean()), "mel_std": float(x.std()),
             "source": f"train split, {args.n_frames // 1_000_000}M random frames, seed {args.seed}"}
    print(json.dumps(stats, indent=1))
    if not args.dry_run:
        json.dump(stats, open(os.path.join(args.cache_dir, "stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
