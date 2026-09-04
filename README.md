# drum-jepa

An action-conditioned JEPA world model of drum sound, built on
[E-GMD](https://magenta.withgoogle.com/datasets/e-gmd) and following the
Music-JEPA design (Wang, Fang, LeCun 2026, arXiv 2607.22000): drum audio is
the state, the drumroll plus hi-hat pedal position is the action, and the
model predicts the next audio state in latent space.

The architecture is reimplemented from the paper. Scaffolding comes from
[I-JEPA](https://github.com/facebookresearch/ijepa) (masked predictor, EMA
target encoder), [LeWorldModel](https://github.com/lucas-maes/le-wm)
(action conditioning), and [audio2midi](https://github.com/ZZWaang/audio2midi)
(spectrogram conventions). See `docs/experiments.md` for the research plan.

## Setup

```bash
scripts/setup_env.sh
source .venv/bin/activate
```

## Download the data

Run the downloads under `caffeinate` so they continue when the lid is closed.
The script resumes partial files and verifies checksums.

```bash
# MIDI archive (103 MB), metadata CSV, and the upstream repos
caffeinate -i scripts/download.sh

# Audio archive (90 GB zip). Put it on an external drive if needed:
DATA_DIR=/Volumes/ext/egmd caffeinate -i scripts/download.sh --audio
```

Don't unpack the audio zip. Choose kits first, then extract only those:

```bash
python scripts/list_kits.py            # 43 kits with hours; pick 6 train + 8 held-out
# edit configs/kits_v1.yaml
python scripts/extract_kits.py          # pulls only the selected WAVs
```

## Inventory the MIDI

Run this before any modeling. It answers which pitches occur, whether
cymbal chokes (polyphonic aftertouch) survived, whether the hi-hat pedal
signal is continuous or bimodal, and how dense 2-second windows are.

```bash
python scripts/inventory.py
```

Commit `docs/inventory.md`. If it reports unmapped pitches, add them to
`drumjepa/drum_map.py` and bump `MAP_VERSION`.

## Layout

```
configs/        kit split (kits_v1.yaml)
drumjepa/       package: drum_map.py (pitch -> class), model code to come
scripts/        download, inventory, kit selection, extraction
docs/           inventory tables, experiment plan
notes/          MPS workarounds and other running notes
data/           downloads and caches (ignored by git)
third_party/    cloned upstream code (ignored by git)
```

## Drum map

`drumjepa/drum_map.py` defines `DRUM_MAP_V1` (14 classes: kick, snare,
rimshot, cross-stick, three toms, closed/open/pedal hi-hat, crash 1, crash 2,
ride, ride bell) and the Magenta 9-class projection for comparison with the
ADT literature. Every collapse of two pitches makes the audio less determined
by the action, so timbrally distinct zones stay separate. The map version is
recorded in every cached tensor and run config.
