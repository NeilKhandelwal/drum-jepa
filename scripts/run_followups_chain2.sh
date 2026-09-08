#!/usr/bin/env bash
# docs/followups.md item 2, seeds 1 and 2 of the mel control. Waits for the first
# chain (runs/followups_chain.log) to print CHAIN_DONE, then trains and evaluates
# each seed in turn. Launch:
#   caffeinate -i nohup bash scripts/run_followups_chain2.sh > runs/followups_chain2.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ACTION_ONLY=runs/drumjepa_v1_actiononly
stage () { echo "STAGE_DONE $1 $(date)"; }
evals () {
    local run=$1
    $PY scripts/eval_e1.py --run-dir "$run" > "$run/e1.log" 2>&1;  stage "E1 $run"
    $PY scripts/eval_e2.py --full "$run" --action-only "$ACTION_ONLY" > "$run/e2.log" 2>&1;  stage "E2 $run"
    $PY scripts/eval_e5.py --drumjepa "$run" --skip aojepa > "$run/e5.log" 2>&1;  stage "E5 $run"
    $PY scripts/eval_readout.py --run-dir "$run" --split test > "$run/readout.log" 2>&1;  stage "READOUT $run"
}
echo "CHAIN2_WAITING $(date)"
until grep -q CHAIN_DONE runs/followups_chain.log; do sleep 120; done
echo "CHAIN2_START $(date)"
for s in 1 2; do
    $PY scripts/train.py --config configs/drumjepa_v1_melrec_s$s.yaml > runs/drumjepa_v1_melrec_s$s.log 2>&1
    stage "TRAIN runs/drumjepa_v1_melrec_s$s"
    evals runs/drumjepa_v1_melrec_s$s
done
echo "CHAIN2_DONE $(date)"
