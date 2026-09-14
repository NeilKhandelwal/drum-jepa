#!/usr/bin/env bash
# The variance-covariance control (docs/followups.md, "Open"): aux_target "vc"
# at aux_rec 0.1, three seeds, with the same eval block as the mel control
# (scripts/run_followups_chain2.sh) so the three aux terms are compared on
# identical evals: E1, E2 against the v1 action-only run, E5 without AO-JEPA,
# readout on test.
#
#   caffeinate -i nohup bash scripts/run_vc_chain.sh > runs/vc_chain.log 2>&1 &
#
# Sequential: three ~1 h 17 min trainings plus ~14 min of evals each, ~4 h 35 min
# total. Progress: grep -E '^(STAGE_DONE|VC_CHAIN_)' runs/vc_chain.log. set -e,
# so the first failure stops the chain and the last STAGE_DONE says how far it
# got. Plug the charger in: caffeinate -i does not stop low-battery sleep.
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

echo "VC_CHAIN_START $(date)"
for cfg in drumjepa_v1_vcrec drumjepa_v1_vcrec_s1 drumjepa_v1_vcrec_s2; do
    $PY scripts/train.py --config configs/$cfg.yaml > runs/$cfg.log 2>&1
    stage "TRAIN runs/$cfg"
    evals runs/$cfg
done
echo "VC_CHAIN_DONE $(date)"
