#!/bin/bash
# Full-Scale E2 (Joint-Training, tau=0.2, Geometry=center_subproto, Solver=pinv)
# Purpose: Formal validation of the V2 mainline across all 21 datasets.
# Strategy: Joint training (NO --detach-local-latent), tau=0.2 fixed, 100 epochs.
# NOTE: Using 'pinv' solver and 'center_subproto' geometry as they are the 
# performance leaders from diagnostic sweeps.

PYTHON=/home/THL/miniconda3/envs/pia/bin/python
SCRIPTPATH="scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py"
export CUDA_VISIBLE_DEVICES=0

OUT_ROOT="out/_active/verify_e2_tau02_fullscale_20260414"
SEED=1
EPOCHS=100
TAU=0.2

DATASETS=(
    "har"
    "natops"
    "fingermovements"
    "selfregulationscp1"
    "basicmotions"
    "handmovementdirection"
    "uwavegesturelibrary"
    "epilepsy"
    "atrialfibrillation"
    "pendigits"
    "racketsports"
    "articularywordrecognition"
    "heartbeat"
    "selfregulationscp2"
    "libras"
    "japanesevowels"
    "cricket"
    "handwriting"
    "ering"
    "motorimagery"
    "ethanolconcentration"
)

echo "=========================================================="
echo "  Full-Scale E2 (tau=$TAU) Corrected Sweep (pinv + center_sub) "
echo "  Target: ${#DATASETS[@]} datasets, seed=$SEED            "
echo "=========================================================="

mkdir -p $OUT_ROOT

for DS in "${DATASETS[@]}"; do
    RUN_TAG="e2_tau${TAU}_${DS}_v2Corrected"
    OUT_DIR="$OUT_ROOT/e2/$RUN_TAG"

    # Skip if already complete
    if [ -f "$OUT_DIR/summary.json" ]; then
        echo "Skipping $DS (already complete)"
        continue
    fi

    echo "Running E2 tau=$TAU (pinv+center_sub) for $DS..."
    $PYTHON $SCRIPTPATH \
        --dataset $DS \
        --arm e2 \
        --epochs $EPOCHS \
        --seed $SEED \
        --subproto-temperature $TAU \
        --closed-form-solve-mode pinv \
        --prototype-geometry-mode center_subproto \
        --dataflow-probe \
        --out-root $OUT_ROOT \
        --run-tag "$RUN_TAG"
done

echo "=========================================================="
echo "  Full-Scale Corrected Sweep Complete!                  "
echo "=========================================================="
