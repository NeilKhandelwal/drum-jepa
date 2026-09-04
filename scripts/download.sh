#!/usr/bin/env bash
# Resumable download of E-GMD plus the upstream code this repo borrows from.
#
# Usage:
#   caffeinate -i scripts/download.sh            # MIDI + metadata + code (small)
#   caffeinate -i scripts/download.sh --audio    # also the 90 GB audio archive
#
# Re-running resumes any partial file. Checksums are verified after download.
# Set DATA_DIR to put the archives somewhere other than ./data (for example an
# external drive): DATA_DIR=/Volumes/ext/egmd scripts/download.sh --audio
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
THIRD_PARTY="$REPO_ROOT/third_party"
BASE="https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0"

MIDI_ZIP="e-gmd-v1.0.0-midi.zip"
MIDI_SHA="5e70a6f4d760385a5e5ec986a2f02179d96f61181a920e592876b577a75844d3"
AUDIO_ZIP="e-gmd-v1.0.0.zip"
AUDIO_SHA="7d9a264fb4c9eabd9fec09d5f8e333192f529b1a1b845d170279a977ac436053"
META_CSV="e-gmd-v1.0.0.csv"

WANT_AUDIO=0
for arg in "$@"; do
  case "$arg" in
    --audio) WANT_AUDIO=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$DATA_DIR" "$THIRD_PARTY"

fetch() {  # fetch <url> <dest>
  echo "==> $2"
  curl -L --fail --retry 20 --retry-delay 15 --retry-all-errors \
       -C - -o "$2" "$1"
}

verify() {  # verify <file> <sha256>
  echo "==> verifying $(basename "$1")"
  local got
  got="$(shasum -a 256 "$1" | awk '{print $1}')"
  if [[ "$got" != "$2" ]]; then
    echo "CHECKSUM MISMATCH for $1" >&2
    echo "  expected $2" >&2
    echo "  got      $got" >&2
    echo "Delete the file and re-run to download it again." >&2
    exit 1
  fi
  echo "    ok"
}

# --- metadata + MIDI (small) --------------------------------------------------
fetch "$BASE/$META_CSV" "$DATA_DIR/$META_CSV"
fetch "$BASE/$MIDI_ZIP" "$DATA_DIR/$MIDI_ZIP"
verify "$DATA_DIR/$MIDI_ZIP" "$MIDI_SHA"

if [[ ! -d "$DATA_DIR/midi" ]]; then
  echo "==> extracting MIDI archive"
  mkdir -p "$DATA_DIR/midi"
  unzip -q "$DATA_DIR/$MIDI_ZIP" -d "$DATA_DIR/midi"
fi

# --- upstream code ------------------------------------------------------------
clone() {  # clone <url> <dir>
  if [[ -d "$2/.git" ]]; then
    echo "==> $2 already cloned"
  else
    echo "==> cloning $1"
    git clone --depth 1 "$1" "$2"
  fi
}
clone https://github.com/facebookresearch/ijepa.git   "$THIRD_PARTY/ijepa"
clone https://github.com/lucas-maes/le-wm.git         "$THIRD_PARTY/le-wm"
clone https://github.com/ZZWaang/audio2midi.git        "$THIRD_PARTY/audio2midi"

# --- audio (90 GB zip, 132 GB extracted) --------------------------------------
if [[ "$WANT_AUDIO" == "1" ]]; then
  avail_gb=$(df -g "$DATA_DIR" | awk 'NR==2{print $4}')
  echo "==> free space at $DATA_DIR: ${avail_gb} GB (need ~95 GB for the zip alone)"
  if [[ "$avail_gb" -lt 95 ]]; then
    echo "Not enough space. Set DATA_DIR to an external drive." >&2
    exit 1
  fi
  fetch "$BASE/$AUDIO_ZIP" "$DATA_DIR/$AUDIO_ZIP"
  verify "$DATA_DIR/$AUDIO_ZIP" "$AUDIO_SHA"
  echo
  echo "Audio archive downloaded. Do NOT unzip it fully (132 GB)."
  echo "Pick kits in configs/kits_v1.yaml, then run:"
  echo "  python scripts/extract_kits.py"
fi

echo "==> done"
