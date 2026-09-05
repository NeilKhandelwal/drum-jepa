"""Build the preprocessing cache: log-mel per (sequence, kit), action per sequence.

Layout, per split under data/cache/<cache_version>/<split>/:
  mel.f16        raw float16 memmap, (total_frames, 229); one contiguous block per file
  mel_index.csv  file_idx, seq_id, kit_name, kit_id, start, n_frames, audio_filename
  roll.f16       raw float16 memmap, (total_seq_frames, K_V1); one block per sequence
  cc4.f16        raw float16 memmap, (total_seq_frames,)
  seq_index.csv  seq_id, start, n_frames, midi_filename (canonical kit's file)
  meta.json      MAP_VERSION, split_version, kits, mel params, cache_version

n_frames for a sequence = the frame count of its audio (identical across kits,
asserted). Files shorter than one 2 s segment are skipped.

    python scripts/build_cache.py [--splits validation test train] [--workers 6]
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drumjepa import features as F  # noqa: E402
from drumjepa.drum_map import CLASSES_V1, MAP_VERSION  # noqa: E402

CACHE_VERSION = "v1"


def mel_job(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return F.log_mel(y.mean(1), sr).astype(np.float16)


def build_split(df, split, cfg, out, workers, midi_dir, audio_dir):
    os.makedirs(out, exist_ok=True)
    kits = cfg["train_kits"] + cfg["heldout_kits"]
    kit_id = {k: i for i, k in enumerate(kits)}
    d = df[(df.split == split) & df.kit_name.isin(kits)].copy()
    d = d[d.duration >= F.SEG_FRAMES / F.FRAMES_PER_S]
    d = d.sort_values(["id", "kit_name"]).reset_index(drop=True)
    canon = df[(df.split == split) & (df.kit_name == cfg["canonical_kit"])].set_index("id")
    print(f"[{split}] {len(d)} files, {d.id.nunique()} sequences, {len(kits)} kits")

    # --- mel per (sequence, kit) ---
    paths = [os.path.join(audio_dir, f) for f in d.audio_filename]
    n_frames_by_seq = {}
    start = 0
    with Pool(workers) as pool, open(os.path.join(out, "mel.f16"), "wb") as fmel, \
            open(os.path.join(out, "mel_index.csv"), "w", newline="") as fidx:
        w = csv.writer(fidx)
        w.writerow(["file_idx", "seq_id", "kit_name", "kit_id", "start", "n_frames", "audio_filename"])
        for i, (mel, row) in enumerate(zip(tqdm(pool.imap(mel_job, paths, chunksize=4),
                                                 total=len(paths), desc=f"mel/{split}"),
                                            d.itertuples())):
            n = mel.shape[0]
            prev = n_frames_by_seq.setdefault(row.id, n)
            assert prev == n, f"frame count differs across kits for {row.id}: {prev} vs {n}"
            fmel.write(mel.tobytes())
            w.writerow([i, row.id, row.kit_name, kit_id[row.kit_name], start, n, row.audio_filename])
            start += n
    total_mel_frames = start

    # --- action per sequence (canonical MIDI) ---
    start = 0
    with open(os.path.join(out, "roll.f16"), "wb") as froll, open(os.path.join(out, "cc4.f16"), "wb") as fcc, \
            open(os.path.join(out, "seq_index.csv"), "w", newline="") as fidx:
        w = csv.writer(fidx)
        w.writerow(["seq_id", "start", "n_frames", "midi_filename"])
        for seq_id, n in tqdm(n_frames_by_seq.items(), desc=f"action/{split}"):
            mf = canon.loc[seq_id, "midi_filename"]
            roll, cc4 = F.midi_to_action(os.path.join(midi_dir, mf), n)
            froll.write(roll.astype(np.float16).tobytes())
            fcc.write(cc4.astype(np.float16).tobytes())
            w.writerow([seq_id, start, n, mf])
            start += n

    meta = dict(cache_version=CACHE_VERSION, MAP_VERSION=MAP_VERSION, split_version=cfg["split_version"],
                split=split, kits=kits, canonical_kit=cfg["canonical_kit"], classes=CLASSES_V1,
                mel=F.MEL_PARAMS, seg_frames=F.SEG_FRAMES, velocity_scaling="velocity/127",
                cc4_scaling="value/127, forward-filled per frame, back-filled before first message",
                n_files=len(d), n_sequences=len(n_frames_by_seq), mel_frames=total_mel_frames,
                seq_frames=start, dtype="float16")
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"[{split}] done: {total_mel_frames:,} mel frames, {start:,} action frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/e-gmd-v1.0.0.csv")
    ap.add_argument("--kits", default="configs/kits_v1.yaml")
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--midi-dir", default="data/midi/e-gmd-v1.0.0")
    ap.add_argument("--out", default=f"data/cache/{CACHE_VERSION}")
    ap.add_argument("--splits", nargs="+", default=["validation", "test", "train"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N sequences per split")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.kits))
    df = pd.read_csv(args.csv)
    df = df[df.audio_filename.notna()]
    if args.limit:
        keep = df.id.drop_duplicates().groupby(df.split).head(args.limit)
        df = df[df.id.isin(keep)]
    for split in args.splits:
        build_split(df, split, cfg, os.path.join(args.out, split), args.workers,
                    args.midi_dir, args.audio_dir)


if __name__ == "__main__":
    main()
