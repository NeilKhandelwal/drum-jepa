#!/usr/bin/env bash
# Create a virtualenv with PyTorch (MPS on Apple silicon) and the repo deps.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python - << 'PY'
import torch
print("torch", torch.__version__, "| mps available:", torch.backends.mps.is_available())
PY
echo "Activate with: source .venv/bin/activate"
