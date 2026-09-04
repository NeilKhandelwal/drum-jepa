"""Extract only the WAVs for the kits listed in configs/kits_v1.yaml from the
E-GMD audio zip, without unpacking the whole 132 GB archive.

    python scripts/extract_kits.py [--zip data/e-gmd-v1.0.0.zip] [--out data/audio]
"""
import argparse
import os
import zipfile

import pandas as pd
import yaml
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/e-gmd-v1.0.0.zip")
    ap.add_argument("--csv", default="data/e-gmd-v1.0.0.csv")
    ap.add_argument("--kits", default="configs/kits_v1.yaml")
    ap.add_argument("--out", default="data/audio")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.kits))
    kits = set(cfg["train_kits"]) | set(cfg["heldout_kits"])
    if not kits:
        raise SystemExit("configs/kits_v1.yaml has no kits listed. Run scripts/list_kits.py first.")

    df = pd.read_csv(args.csv)
    df = df[df["kit_name"].isin(kits) & df["audio_filename"].notna()]
    wanted = set(df["audio_filename"])
    print(f"{len(wanted)} audio files across {len(kits)} kits")

    os.makedirs(args.out, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        names = zf.namelist()
        # archive entries are usually prefixed with the top-level folder
        by_suffix = {}
        for n in names:
            for w in wanted:
                if n.endswith(w):
                    by_suffix[w] = n
                    break
        missing = wanted - set(by_suffix)
        if missing:
            print(f"WARNING: {len(missing)} wanted files not found in zip "
                  f"(known E-GMD recording errors); first few: {sorted(missing)[:3]}")
        for w, n in tqdm(by_suffix.items(), desc="extracting"):
            dest = os.path.join(args.out, w)
            if os.path.exists(dest):
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(n) as src, open(dest, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
