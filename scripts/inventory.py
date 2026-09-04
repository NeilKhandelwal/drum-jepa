"""Inventory of the E-GMD MIDI: what the action data actually contains.

Answers, before any modeling:
  1. Which MIDI pitches occur, how often, and whether drum_map.py covers them.
  2. Whether cymbal chokes survived (polyphonic aftertouch messages).
  3. Whether hi-hat pedal position (CC 4) is a continuous signal or bimodal.
  4. Onset density per 2-second window (for stratified sampling).
  5. Velocity distribution per class.

E-GMD ships one MIDI file per (sequence, kit) pair, so most files are
byte-identical duplicates. Files are deduplicated by content hash.

    python scripts/inventory.py [--midi-dir data/midi] [--workers 8]

Writes data/inventory.json (raw) and docs/inventory.md (tables to commit).
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import mido
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drumjepa.drum_map import (DRUM_MAP_V1, PITCH_DESCRIPTION, CLASSES_V1,  # noqa: E402
                               HH_POSITION_CC, MAP_VERSION)

WINDOW_S = 2.0
DENSITY_BINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10**9)]
VEL_BIN = 8  # 16 velocity bins


def density_bin(n):
    for i, (lo, hi) in enumerate(DENSITY_BINS):
        if lo <= n <= hi:
            return i
    return len(DENSITY_BINS) - 1


def scan(path):
    """Scan one MIDI file. Returns plain dicts (picklable)."""
    out = {
        "notes": Counter(),          # pitch -> count
        "vel": Counter(),            # (class, vel_bin) -> count
        "polytouch": Counter(),      # pitch -> count
        "aftertouch": 0,             # channel aftertouch count
        "cc": Counter(),             # cc number -> count
        "cc4_hist": Counter(),       # value -> count
        "density": Counter(),        # density bin -> windows
        "duration": 0.0,
        "unmapped": Counter(),
    }
    try:
        mf = mido.MidiFile(path)
    except Exception as e:  # corrupt file
        out["error"] = repr(e)
        return out

    t = 0.0
    windows = Counter()
    for msg in mf:  # iterating a MidiFile yields delta times in seconds
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            out["notes"][msg.note] += 1
            cls = DRUM_MAP_V1.get(msg.note)
            if cls is None:
                out["unmapped"][msg.note] += 1
            else:
                out["vel"][(cls, msg.velocity // VEL_BIN)] += 1
            windows[int(t // WINDOW_S)] += 1
        elif msg.type == "polytouch":
            out["polytouch"][msg.note] += 1
        elif msg.type == "aftertouch":
            out["aftertouch"] += 1
        elif msg.type == "control_change":
            out["cc"][msg.control] += 1
            if msg.control == HH_POSITION_CC:
                out["cc4_hist"][msg.value] += 1
    out["duration"] = t
    n_windows = int(t // WINDOW_S) + 1
    for w in range(n_windows):
        out["density"][density_bin(windows.get(w, 0))] += 1
    # Counters with tuple keys don't survive JSON; stringify later.
    return out


def file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi-dir", default="data/midi")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--json", default="data/inventory.json")
    ap.add_argument("--md", default="docs/inventory.md")
    args = ap.parse_args()

    paths = []
    for root, _, files in os.walk(args.midi_dir):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                paths.append(os.path.join(root, f))
    if not paths:
        raise SystemExit(f"no MIDI files under {args.midi_dir}; run scripts/download.sh first")

    # dedupe by content
    seen, unique = set(), []
    for p in tqdm(paths, desc="hashing"):
        h = file_hash(p)
        if h not in seen:
            seen.add(h)
            unique.append(p)
    print(f"{len(paths)} files, {len(unique)} unique by content")

    agg = {k: Counter() for k in ("notes", "vel", "polytouch", "cc", "cc4_hist", "density", "unmapped")}
    agg_after, agg_dur, errors = 0, 0.0, []
    with Pool(args.workers) as pool:
        for r in tqdm(pool.imap_unordered(scan, unique, chunksize=8), total=len(unique), desc="scanning"):
            if "error" in r:
                errors.append(r["error"])
                continue
            for k in agg:
                agg[k].update(r[k])
            agg_after += r["aftertouch"]
            agg_dur += r["duration"]

    # ---- assemble report ----
    total_notes = sum(agg["notes"].values())
    class_counts = Counter()
    for p, n in agg["notes"].items():
        class_counts[DRUM_MAP_V1.get(p, "UNMAPPED")] += n

    cc4 = agg["cc4_hist"]
    cc4_total = sum(cc4.values())
    cc4_low = sum(v for k, v in cc4.items() if k <= 15)
    cc4_high = sum(v for k, v in cc4.items() if k >= 112)
    cc4_mid = cc4_total - cc4_low - cc4_high

    report = {
        "map_version": MAP_VERSION,
        "files_total": len(paths),
        "files_unique": len(unique),
        "errors": len(errors),
        "unique_hours": agg_dur / 3600,
        "total_note_ons": total_notes,
        "pitch_counts": {int(p): n for p, n in sorted(agg["notes"].items(), key=lambda x: -x[1])},
        "unmapped_pitches": {int(p): n for p, n in agg["unmapped"].items()},
        "class_counts_v1": {c: class_counts.get(c, 0) for c in CLASSES_V1},
        "polytouch_by_pitch": {int(p): n for p, n in agg["polytouch"].items()},
        "channel_aftertouch": agg_after,
        "cc_counts": {int(c): n for c, n in agg["cc"].items()},
        "cc4": {"total": cc4_total, "low_0_15": cc4_low, "mid_16_111": cc4_mid, "high_112_127": cc4_high,
                "hist": {int(k): v for k, v in sorted(cc4.items())}},
        "density_2s": {f"{lo}-{hi if hi < 10**9 else 'inf'}": agg["density"].get(i, 0)
                       for i, (lo, hi) in enumerate(DENSITY_BINS)},
        "velocity_by_class": {c: [agg["vel"].get((c, b), 0) for b in range(128 // VEL_BIN)] for c in CLASSES_V1},
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(report, f, indent=2)

    # ---- markdown ----
    L = []
    L.append(f"# E-GMD MIDI inventory (drum map {MAP_VERSION})\n")
    L.append(f"- Files: {len(paths)} total, {len(unique)} unique by content, {len(errors)} unreadable")
    L.append(f"- Unique duration: {agg_dur/3600:.1f} h; note-ons: {total_notes:,}\n")

    L.append("## Pitches\n")
    L.append("| pitch | count | share | V1 class | GMD description |")
    L.append("|---:|---:|---:|---|---|")
    for p, n in sorted(agg["notes"].items(), key=lambda x: -x[1]):
        L.append(f"| {p} | {n:,} | {100*n/total_notes:.2f}% | {DRUM_MAP_V1.get(p, '**UNMAPPED**')} | {PITCH_DESCRIPTION.get(p, '')} |")
    if agg["unmapped"]:
        L.append(f"\n**{len(agg['unmapped'])} unmapped pitches** — add to drum_map.py or justify dropping.\n")

    L.append("\n## Classes (V1)\n")
    L.append("| class | count | share |")
    L.append("|---|---:|---:|")
    for c in CLASSES_V1:
        n = class_counts.get(c, 0)
        L.append(f"| {c} | {n:,} | {100*n/max(total_notes,1):.2f}% |")

    L.append("\n## Chokes (aftertouch)\n")
    if agg["polytouch"]:
        L.append("Polyphonic aftertouch present — chokes survived.\n")
        L.append("| pitch | events |")
        L.append("|---:|---:|")
        for p, n in sorted(agg["polytouch"].items(), key=lambda x: -x[1]):
            L.append(f"| {p} ({DRUM_MAP_V1.get(p,'?')}) | {n:,} |")
    else:
        L.append("No polyphonic aftertouch found. Chokes are not encoded; the choke eval is off the table.")
    L.append(f"\nChannel aftertouch messages: {agg_after:,}\n")

    L.append("## Hi-hat pedal position (CC 4)\n")
    if cc4_total:
        L.append(f"- events: {cc4_total:,}")
        L.append(f"- value 0-15: {100*cc4_low/cc4_total:.1f}%  |  16-111: {100*cc4_mid/cc4_total:.1f}%  |  112-127: {100*cc4_high/cc4_total:.1f}%")
        verdict = ("BIMODAL — treat as a near-binary open/closed signal, not a continuous control variable."
                   if cc4_mid / cc4_total < 0.15 else
                   "Substantial mid-range mass — position may carry continuous information; inspect the histogram.")
        L.append(f"- verdict: {verdict}\n")
        L.append("Other CC numbers seen: " + ", ".join(f"{c}: {n:,}" for c, n in sorted(agg["cc"].items())))
    else:
        L.append("No CC 4 messages found.")

    L.append("\n## Onset density per 2 s window\n")
    L.append("| onsets in window | windows | share |")
    L.append("|---|---:|---:|")
    tot_w = sum(agg["density"].values())
    for i, (lo, hi) in enumerate(DENSITY_BINS):
        n = agg["density"].get(i, 0)
        L.append(f"| {lo}-{hi if hi < 10**9 else '∞'} | {n:,} | {100*n/max(tot_w,1):.1f}% |")

    L.append("\n## Velocity by class (16 bins of 8)\n")
    L.append("| class | " + " | ".join(str(b*VEL_BIN) for b in range(128 // VEL_BIN)) + " |")
    L.append("|---|" + "---:|" * (128 // VEL_BIN))
    for c in CLASSES_V1:
        row = report["velocity_by_class"][c]
        L.append(f"| {c} | " + " | ".join(f"{v:,}" for v in row) + " |")

    os.makedirs(os.path.dirname(args.md), exist_ok=True)
    with open(args.md, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.json} and {args.md}")
    if agg["unmapped"]:
        print(f"WARNING: unmapped pitches: {dict(agg['unmapped'])}")
    if errors:
        print(f"{len(errors)} unreadable files, e.g. {errors[0]}")


if __name__ == "__main__":
    main()
