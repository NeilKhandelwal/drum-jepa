#!/usr/bin/env bash
# docs/followups.md items 2 (control aux target) and 3 (longer baseline), in order.
#
#   caffeinate -i nohup bash scripts/run_followups_chain.sh > runs/followups_chain.log 2>&1 &
#
# Sequential: two ~75 min trainings plus three eval blocks, ~5-6 h total. Progress:
# grep -E '^(STAGE_DONE|CHAIN_DONE)' runs/followups_chain.log. set -e, so the first
# failure stops the chain and the last STAGE_DONE line says how far it got.
#
# Eval arguments are the ones the seed runs used (runs/drumjepa_v1_auxrec01_s1/e*.log):
# E1 and E2 on validation with K=4, E2 against the shared action-only baseline, E5
# with AO-JEPA skipped (the control and the baseline are compared to random and raw
# mel, not to AO-JEPA).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ACTION_ONLY=runs/drumjepa_v1_actiononly

stage () { echo "STAGE_DONE $1 $(date)"; }

evals () {  # E1, E2, E5 on one run directory, logs next to the results
    local run=$1
    $PY scripts/eval_e1.py --run-dir "$run" > "$run/e1.log" 2>&1
    stage "E1 $run"
    $PY scripts/eval_e2.py --full "$run" --action-only "$ACTION_ONLY" > "$run/e2.log" 2>&1
    stage "E2 $run"
    $PY scripts/eval_e5.py --drumjepa "$run" --skip aojepa > "$run/e5.log" 2>&1
    stage "E5 $run"
    $PY scripts/eval_readout.py --run-dir "$run" --split test > "$run/readout.log" 2>&1
    stage "READOUT $run"
}

echo "CHAIN_START $(date)"

# 1. Item 2: the mel control, 20 epochs, seed 0.
$PY scripts/train.py --config configs/drumjepa_v1_melrec.yaml > runs/drumjepa_v1_melrec.log 2>&1
stage "TRAIN runs/drumjepa_v1_melrec"

# 2. Its evals.
evals runs/drumjepa_v1_melrec

# 3. Item 3: the 40-epoch baseline, seed 0. keep_epochs [19] leaves epoch19.pt behind.
$PY scripts/train.py --config configs/drumjepa_v1_e40.yaml > runs/drumjepa_v1_e40.log 2>&1
stage "TRAIN runs/drumjepa_v1_e40"

# 4. The epoch-19 snapshot as a sibling run directory. eval_e1/e2/e5 read exactly
# config.json and last.pt from a run directory, so a copy plus a symlink is enough;
# each eval writes its own subdirectory here and leaves the e40 results alone.
mkdir -p runs/drumjepa_v1_e40_at20
cp runs/drumjepa_v1_e40/config.json runs/drumjepa_v1_e40_at20/config.json
ln -sf ../drumjepa_v1_e40/epoch19.pt runs/drumjepa_v1_e40_at20/last.pt
stage "SNAPSHOT runs/drumjepa_v1_e40_at20"
evals runs/drumjepa_v1_e40_at20

# 5. The 40-epoch end state.
evals runs/drumjepa_v1_e40

echo "CHAIN_DONE $(date)"
