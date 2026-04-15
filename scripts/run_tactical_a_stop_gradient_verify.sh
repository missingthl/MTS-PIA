#!/bin/bash
# V2 Controlled Entry: Tactical A (Stop-Gradient / Isolation)
# Purpose: Prevent local head from polluting the backbone, especially for UWave.

PYTHON=/home/THL/miniconda3/envs/pia/bin/python
SCRIPTPATH="scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py"
export CUDA_VISIBLE_DEVICES=1

# Use the peak tau discovered in the sweep
TAU=0.2
DATASETS=("natops" "fingermovements" "selfregulationscp1" "uwavegesturelibrary")
OUT_ROOT="out/_active/verify_resnet1d_tactical_a_20260414"

echo "=========================================================="
echo "  V2 Controlled Entry: Tactical A (Stop-Gradient)         "
echo "  Testing tau=$TAU with --detach-local-latent             "
echo "=========================================================="

mkdir -p $OUT_ROOT

for DS in "${DATASETS[@]}"; do
    echo "Running Tactical A for $DS..."
    $PYTHON $SCRIPTPATH \
        --dataset $DS \
        --arm e2 \
        --epochs 100 \
        --seed 1 \
        --subproto-temperature $TAU \
        --detach-local-latent \
        --dataflow-probe \
        --out-root $OUT_ROOT \
        --run-tag "tactical_a_tau${TAU}_${DS}"
done

echo "=========================================================="
echo "  Tactical A Experiments Completed!                       "
echo "=========================================================="
