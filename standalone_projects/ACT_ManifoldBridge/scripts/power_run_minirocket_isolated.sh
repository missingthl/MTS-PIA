#!/bin/bash

# Configuration
BACKBONE="minirocket"
SEEDS="1,2,3"
KERNELS=4000
PYTHON="/home/THL/miniconda3/envs/pia/bin/python"
SCRIPT="scripts/run_external_baselines_phase1.py"

DATASETS=("handwriting" "uwavegesturelibrary" "ering" "motorimagery" "natops" "epilepsy" "articularywordrecognition" "har" "japanesevowels" "pendigits" "basicmotions" "cricket" "racketsports" "ethanolconcentration" "libras" "heartbeat" "fingermovements" "selfregulationscp2" "atrialfibrillation" "handmovementdirection")

# Run Core Matrix (Isolated Directories)
CORE_OUT="results/minirocket_final20_core_v1"
for DS in "${DATASETS[@]}"; do
    DS_ROOT="${CORE_OUT}/${DS}"
    mkdir -p "$DS_ROOT"
    echo "Launching Core for $DS..."
    $PYTHON $SCRIPT --backbone $BACKBONE --n-kernels $KERNELS --out-root "$DS_ROOT" --datasets $DS --seeds $SEEDS --arms no_aug,csta_topk_uniform_top5 --pia-gamma 0.1 --eta-safe 0.75 --multiplier 10 --k-dir 10 --group-size 1 >> "${DS_ROOT}/run.log" 2>&1 &
done

# Run Batch B Matrix (Isolated Directories)
BATCHB_OUT="results/minirocket_final20_batchB_v1"
for DS in "${DATASETS[@]}"; do
    DS_ROOT="${BATCHB_OUT}/${DS}"
    mkdir -p "$DS_ROOT"
    echo "Launching Batch B for $DS..."
    $PYTHON $SCRIPT --backbone $BACKBONE --n-kernels $KERNELS --out-root "$DS_ROOT" --datasets $DS --seeds $SEEDS --arms random_cov_state,pca_cov_state --group-size 1 >> "${DS_ROOT}/run.log" 2>&1 &
done

echo "Total 40 dataset-isolated tasks launched."
