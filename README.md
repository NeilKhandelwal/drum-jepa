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

## Build the cache, train, evaluate

Preprocessing turns each extracted kit into memory-mapped log-mel, drumroll,
and hi-hat tracks under `data/cache/v1`. Actions come from the canonical MIDI
of each sequence, never the kit's own file (see `docs/inventory.md`).

```bash
python scripts/build_cache.py            # ~1 h for 14 kits; skips built splits
python scripts/bench_loader.py           # check the loader is not the bottleneck
```

Training follows the Music-JEPA recipe in `configs/drumjepa_v1.yaml`. A run
writes `config.json`, `metrics.csv` (both loss terms and the embedding-std
collapse monitors, every epoch), and `last.pt` to its run directory.

```bash
python scripts/train.py --overfit 32 --epochs 300      # wiring test: loss -> ~0
python scripts/train.py                                # 20 epochs on 6 kits, ~75 min on M5 Max
python scripts/train.py --run-dir runs/drumjepa_v1 --resume
python scripts/train.py --config configs/drumjepa_v1_actiononly.yaml   # E2 baseline
python scripts/train.py --config configs/drumjepa_v1_aojepa.yaml       # E5 baseline
```

Each experiment block has one script that writes `results.md`, `results.json`,
and a figure into the run directory; checked-in copies live under `docs/e1`
to `docs/e5`, and `docs/experiments.md` has the verdicts.

```bash
python scripts/eval_e1.py --run-dir runs/drumjepa_v1                  # perturbation win rates
python scripts/eval_e2.py --full runs/drumjepa_v1 --action-only runs/drumjepa_v1_actiononly
python scripts/eval_e3.py --run-dir runs/drumjepa_v1                  # kit-latent probes and geometry
python scripts/eval_e4.py                                             # inverse model (trains a small decoder)
python scripts/eval_e5.py                                             # content probes vs AO-JEPA
python -m pytest tests -q
```

Every probe number is reported next to two controls, a random-init encoder and
the raw log-mel, and only counts if the trained encoder beats both. Decisions
the paper leaves open, and the caveat attached to each, are in
`notes/decisions.md`.

## Layout

```
configs/        kit split (kits_v1.yaml) and run configs (drumjepa_v1*.yaml)
drumjepa/       package: drum_map, features, dataset, model (Es/Ea/f/g), inverse
scripts/        download, inventory, extraction, build_cache, train, eval_e1..e5
tests/          synthetic-batch tests for the model and each eval
docs/           inventory, experiment plan and verdicts, per-block results, followups
notes/          decisions on unspecified details, MPS workarounds
runs/           run directories (ignored by git)
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
