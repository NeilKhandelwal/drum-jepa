"""How each E-GMD kit's MIDI deviates from the canonical performance.

E-GMD re-records every performance through the TD-17 on each kit, and the
shipped MIDI reflects that kit's pad-to-note assignment. Some kits remap pads
(hi-hat -> 54 tambourine, tom rim -> 56 cowbell, rim -> 39 clap) and some drop
hits outright (x-stick on 808/909). This script measures that per kit, so the
"same action, different kit" counterfactual can be trusted where it holds.

Canonical MIDI for a sequence = the pitch histogram shared by the most kits.

    python scripts/kit_midi_diff.py [--midi-dir data/midi/e-gmd-v1.0.0]

Writes data/kit_midi_diff.json and appends a section to docs/inventory.md.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import mido
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drumjepa.drum_map import DRUM_MAP_V1, PITCH_DESCRIPTION  # noqa: E402


def scan(path):
    c = Counter()
    try:
        for m in mido.MidiFile(path):
            if m.type == "note_on" and m.velocity > 0:
                c[m.note] += 1
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    return dict(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/e-gmd-v1.0.0.csv")
    ap.add_argument("--midi-dir", default="data/midi/e-gmd-v1.0.0")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df.audio_filename.notna()].reset_index(drop=True)
    paths = [os.path.join(args.midi_dir, f) for f in df.midi_filename]
    with Pool(args.workers) as pool:
        hists = list(tqdm(pool.imap(scan, paths, chunksize=64), total=len(paths)))
    df["pitch_hist"] = hists

    per_kit = defaultdict(lambda: {"seqs": 0, "canonical": 0, "hits_canon": 0,
                                   "hits_kept": 0, "extra": Counter(),
                                   "missing": Counter(), "canon_by_pitch": Counter()})
    for _, g in df.groupby("id"):
        key = lambda h: tuple(sorted(h.items()))  # noqa: E731
        mode = Counter(key(h) for h in g.pitch_hist).most_common(1)[0][0]
        canon = dict(mode)
        for kit, h in zip(g.kit_name, g.pitch_hist):
            k = per_kit[kit]
            k["seqs"] += 1
            k["canonical"] += int(key(h) == mode)
            for p, n in canon.items():
                k["canon_by_pitch"][p] += n
                k["hits_canon"] += n
                kept = min(h.get(p, 0), n)
                k["hits_kept"] += kept
                k["missing"][p] += n - kept
            for p, n in h.items():
                if p not in canon:
                    k["extra"][p] += n
                elif n > canon[p]:
                    k["extra"][p] += n - canon[p]

    rows = []
    for kit, k in sorted(per_kit.items(), key=lambda kv: -kv[1]["canonical"]):
        missing = {p: n / k["canon_by_pitch"][p] for p, n in k["missing"].items() if n}
        rows.append({
            "kit": kit, "seqs": k["seqs"], "canonical_seqs": k["canonical"],
            "hits_kept_frac": k["hits_kept"] / k["hits_canon"],
            "extra_pitches": dict(sorted(k["extra"].items())),
            "missing_frac_by_pitch": {p: round(f, 3) for p, f in sorted(missing.items())},
        })
    os.makedirs("data", exist_ok=True)
    with open("data/kit_midi_diff.json", "w") as f:
        json.dump(rows, f, indent=1)

    def desc(p):
        return f"{p} {PITCH_DESCRIPTION.get(p, '?')}"

    lines = ["", "## Per-kit MIDI deviation from the canonical performance", "",
             "Canonical = pitch histogram shared by the most kits for that sequence.",
             "`kept` = share of canonical note-ons present at the same pitch on this kit.", "",
             "| kit | canonical seqs | kept | extra pitches (count) | missing (share of that pitch) |",
             "|---|---:|---:|---|---|"]
    for r in rows:
        extra = ", ".join(f"{p} ({n:,})" for p, n in r["extra_pitches"].items()) or "-"
        miss = ", ".join(f"{desc(p)}: {f:.0%}" for p, f in r["missing_frac_by_pitch"].items()
                         if f >= 0.005) or "-"
        lines.append(f"| {r['kit']} | {r['canonical_seqs']}/{r['seqs']} | "
                     f"{r['hits_kept_frac']:.1%} | {extra} | {miss} |")
    with open("docs/inventory.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
