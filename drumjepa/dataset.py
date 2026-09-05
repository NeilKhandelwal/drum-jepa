"""Segment-pair dataset over the preprocessing cache (scripts/build_cache.py).

An item is one transition (x_t, a_{t+1}) -> x_{t+1} on one (sequence, kit):
  x_t, x_t1   log-mel (SEG_FRAMES, 229)
  a_t, a_t1   drumroll (SEG_FRAMES, K_V1), velocity/127
  cc_t, cc_t1 hi-hat pedal position (SEG_FRAMES,), CC4/127
  kit_id, seq_idx, density_bin (onsets in x_t's window, bins from inventory.py)
Segments start every `seg_hop` frames within a file; the pair needs 2 segments.
All 43 kits of a sequence share the action rows, so the memmap index is the
only per-kit state.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from drumjepa.drum_map import K_V1
from drumjepa.features import N_MELS, SEG_FRAMES

DENSITY_EDGES = [0, 1, 3, 6, 11, 21, 41]  # same bins as scripts/inventory.py


def density_bin(n_onsets):
    return int(np.searchsorted(DENSITY_EDGES, n_onsets, side="right") - 1)


class SegmentPairs(Dataset):
    def __init__(self, cache_dir, split, seg_hop=SEG_FRAMES, mel_mean=0.0, mel_std=1.0):
        d = os.path.join(cache_dir, split)
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.mel_mean, self.mel_std = float(mel_mean), float(mel_std)
        self.mel = np.memmap(os.path.join(d, "mel.f16"), np.float16, "r",
                             shape=(self.meta["mel_frames"], N_MELS))
        self.roll = np.memmap(os.path.join(d, "roll.f16"), np.float16, "r",
                              shape=(self.meta["seq_frames"], K_V1))
        self.cc4 = np.memmap(os.path.join(d, "cc4.f16"), np.float16, "r", shape=(self.meta["seq_frames"],))
        files = pd.read_csv(os.path.join(d, "mel_index.csv"))
        seqs = pd.read_csv(os.path.join(d, "seq_index.csv")).reset_index().rename(columns={"index": "seq_idx"})
        files = files.merge(seqs[["seq_id", "seq_idx", "start"]].rename(columns={"start": "seq_start"}), on="seq_id")

        mel_start, seq_start, offset, kit, seq_idx = [], [], [], [], []
        for r in files.itertuples():
            n_seg = (r.n_frames - 2 * SEG_FRAMES) // seg_hop + 1
            for k in range(n_seg):
                mel_start.append(r.start); seq_start.append(r.seq_start); offset.append(k * seg_hop)
                kit.append(r.kit_id); seq_idx.append(r.seq_idx)
        self.mel_start, self.seq_start = np.array(mel_start), np.array(seq_start)
        self.offset, self.kit_id, self.seq_idx = np.array(offset), np.array(kit), np.array(seq_idx)
        a = self.seq_start + self.offset
        self.density = np.array([density_bin((self.roll[s:s + SEG_FRAMES] > 0).sum()) for s in a])
        self.seq_ids = seqs.seq_id.tolist()
        self.kits = self.meta["kits"]

    def __len__(self):
        return len(self.offset)

    def __getitem__(self, i):
        m = self.mel_start[i] + self.offset[i]
        a = self.seq_start[i] + self.offset[i]
        L = SEG_FRAMES
        x = (torch.from_numpy(np.asarray(self.mel[m:m + 2 * L], np.float32)) - self.mel_mean) / self.mel_std
        roll = torch.from_numpy(np.asarray(self.roll[a:a + 2 * L], np.float32))
        cc = torch.from_numpy(np.asarray(self.cc4[a:a + 2 * L], np.float32))
        return {"x_t": x[:L], "x_t1": x[L:], "a_t": roll[:L], "a_t1": roll[L:],
                "cc_t": cc[:L], "cc_t1": cc[L:],
                "kit_id": int(self.kit_id[i]), "seq_idx": int(self.seq_idx[i]),
                "density_bin": int(self.density[i])}


def mel_stats(cache_dir, split="train", max_frames=2_000_000, seed=0):
    """Global log-mel mean/std from a random subset of frames (for normalization)."""
    ds = SegmentPairs(cache_dir, split)
    n = ds.meta["mel_frames"]
    idx = np.sort(np.random.default_rng(seed).choice(n, size=min(max_frames, n), replace=False))
    x = np.asarray(ds.mel[idx], np.float32)
    return float(x.mean()), float(x.std())
