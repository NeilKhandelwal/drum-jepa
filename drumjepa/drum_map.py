"""Roland TD-11/TD-17 MIDI pitch -> drum class mappings for E-GMD.

DRUM_MAP_V1 is the fine-grained map used to build the drumroll action.
K9 is the Magenta 9-class projection, kept for comparison with the ADT
literature (GrooVAE, OaF-Drums).

Every collapse of two pitches into one class makes the audio less
determined by the action. Keep zones separate when they are timbrally
distinct on the kit (rimshot vs head, bell vs bow, crash 1 vs crash 2).
Collapse them only when the sound is nearly the same (tom rim -> tom,
hi-hat edge -> hi-hat).

Pitch numbers follow the Groove MIDI Dataset mapping table and apply
to the CANONICAL MIDI of a sequence (the file shipped for any of the 25
non-remapping kits, e.g. "Acoustic Kit"). Per-kit MIDI files on the
other 18 kits carry pad remaps (54/39/56); see docs/inventory.md. Verify them
against scripts/inventory.py output before training; any pitch seen in
the data but absent here is a bug.

MAP_VERSION is written into every cached tensor and run config. Bump it
if the map changes; results across versions are not comparable.
"""

MAP_VERSION = "v1"

# pitch -> (fine class, GMD description)
_V1 = {
    36: ("kick",       "Kick"),
    38: ("snare",      "Snare (Head)"),
    40: ("rimshot",    "Snare (Rim)"),
    37: ("crossstick", "Snare X-Stick"),
    48: ("tom_hi",     "Tom 1"),
    50: ("tom_hi",     "Tom 1 (Rim)"),
    45: ("tom_mid",    "Tom 2"),
    47: ("tom_mid",    "Tom 2 (Rim)"),
    43: ("tom_lo",     "Tom 3"),
    58: ("tom_lo",     "Tom 3 (Rim)"),
    42: ("hh_closed",  "HH Closed (Bow)"),
    22: ("hh_closed",  "HH Closed (Edge)"),
    46: ("hh_open",    "HH Open (Bow)"),
    26: ("hh_open",    "HH Open (Edge)"),
    44: ("hh_pedal",   "HH Pedal"),
    49: ("crash1",     "Crash 1 (Bow)"),
    55: ("crash1",     "Crash 1 (Edge)"),
    57: ("crash2",     "Crash 2 (Bow)"),
    52: ("crash2",     "Crash 2 (Edge)"),
    51: ("ride",       "Ride (Bow)"),
    59: ("ride",       "Ride (Edge)"),   # "crash on the ride": ride timbre, not crash
    53: ("ride_bell",  "Ride (Bell)"),
}

DRUM_MAP_V1 = {p: c for p, (c, _) in _V1.items()}
PITCH_DESCRIPTION = {p: d for p, (_, d) in _V1.items()}

# Row order of the drumroll. Drums first, then hats, then cymbals.
CLASSES_V1 = [
    "kick", "snare", "rimshot", "crossstick",
    "tom_hi", "tom_mid", "tom_lo",
    "hh_closed", "hh_open", "hh_pedal",
    "crash1", "crash2", "ride", "ride_bell",
]
CLASS_INDEX_V1 = {c: i for i, c in enumerate(CLASSES_V1)}
K_V1 = len(CLASSES_V1)  # 14

# Magenta 9-class projection (GrooVAE / OaF-Drums convention).
K9_CLASSES = ["kick", "snare", "hh_closed", "hh_open",
              "tom_lo", "tom_mid", "tom_hi", "crash", "ride"]
V1_TO_K9 = {
    "kick": "kick",
    "snare": "snare", "rimshot": "snare", "crossstick": "snare",
    "tom_hi": "tom_hi", "tom_mid": "tom_mid", "tom_lo": "tom_lo",
    "hh_closed": "hh_closed", "hh_pedal": "hh_closed", "hh_open": "hh_open",
    "crash1": "crash", "crash2": "crash",
    "ride": "ride", "ride_bell": "ride",
}
K9_INDEX = {c: i for i, c in enumerate(K9_CLASSES)}

# MIDI control-change number for hi-hat pedal position on Roland kits.
HH_POSITION_CC = 4


def pitch_to_class(pitch: int, strict: bool = True):
    """Return the V1 class for a MIDI pitch, or None if unmapped (strict=False)."""
    if pitch in DRUM_MAP_V1:
        return DRUM_MAP_V1[pitch]
    if strict:
        raise KeyError(f"unmapped MIDI pitch {pitch}; add it to drum_map.py")
    return None


def v1_index_to_k9_index():
    """Row-projection vector: K_V1 -> K9 indices."""
    return [K9_INDEX[V1_TO_K9[c]] for c in CLASSES_V1]
