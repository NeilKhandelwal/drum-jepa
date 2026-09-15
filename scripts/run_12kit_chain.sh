#!/usr/bin/env bash
# The 12-train-kit variant (configs/kits_v2.yaml, data/cache/v2): baseline and
# option A (aux_rec 0.1) on 12 kits with v1's 8 held-out kits unchanged. Tests
# whether the held-out content margin closes when training-kit diversity doubles.
#
#   caffeinate -i nohup bash scripts/run_12kit_chain.sh > runs/k12_chain.log 2>&1 &
#
# Sequential: two ~2 h 20 min trainings (~1,240 steps/epoch at batch 128, ~7 min
# per epoch, 20 epochs) plus two eval blocks, ~5-6 h total. Progress:
# grep -E '^(STAGE_DONE|K12_CHAIN_)' runs/k12_chain.log. set -e, so the first
# failure stops the chain and the last STAGE_DONE line says how far it got.
#
# No waiting loop: nothing else has to finish first, unlike
# scripts/run_followups_chain2.sh, which this is modelled on.
#
# E2 IS DELIBERATELY LEFT OUT. eval_e2.py compares a full model against an
# action-only model trained on the SAME cache under the same recipe. The only
# action-only baseline that exists is runs/drumjepa_v1_actiononly, trained on
# data/cache/v1 with v1's normalization stats and 6 kits; running E2 against it
# would compare errors measured in two different mel scales on two different kit
# sets (notes/decisions.md, "Mel normalization" and "Cross-model prediction
# errors"). If E2 is wanted at 12 kits, train a v2 action-only run first
# (a copy of configs/drumjepa_v2_12k.yaml with use_state: false) and add a third
# training stage here.
#
# Eval arguments match the v1 follow-up chains: E1 on validation with K=4, E5
# with AO-JEPA skipped (there is no v2 AO-JEPA either; the run is compared to
# the random-init and raw-mel controls, which are computed inside eval_e5.py
# from the same cache), readout on test. Every eval reads cache_dir and
# train_kits from the run's own config.json, so they follow v2 automatically and
# derive the 8 held-out kits as the cache kits minus the 12 train kits.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

stage () { echo "STAGE_DONE $1 $(date)"; }

evals () {  # E1, E5, readout on one run directory, logs next to the results
    local run=$1
    $PY scripts/eval_e1.py --run-dir "$run" > "$run/e1.log" 2>&1
    stage "E1 $run"
    $PY scripts/eval_e5.py --drumjepa "$run" --skip aojepa > "$run/e5.log" 2>&1
    stage "E5 $run"
    $PY scripts/eval_readout.py --run-dir "$run" --split test > "$run/readout.log" 2>&1
    stage "READOUT $run"
}

echo "K12_CHAIN_START $(date)"

# 1. Baseline: v1 recipe, 12 train kits, seed 0.
$PY scripts/train.py --config configs/drumjepa_v2_12k.yaml > runs/drumjepa_v2_12k.log 2>&1
stage "TRAIN runs/drumjepa_v2_12k"
evals runs/drumjepa_v2_12k

# 2. Option A at the sweep's sweet spot, same 12 kits, seed 0.
$PY scripts/train.py --config configs/drumjepa_v2_12k_auxrec01.yaml > runs/drumjepa_v2_12k_auxrec01.log 2>&1
stage "TRAIN runs/drumjepa_v2_12k_auxrec01"
evals runs/drumjepa_v2_12k_auxrec01

echo "K12_CHAIN_DONE $(date)"
