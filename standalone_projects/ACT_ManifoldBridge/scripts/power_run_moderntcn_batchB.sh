#!/bin/bash

# Configuration
BACKBONE="moderntcn"
SEEDS="1,2,3"
PYTHON="/home/THL/miniconda3/envs/pia/bin/python"
SCRIPT="scripts/run_external_baselines_phase1.py"
OUT_ROOT="results/moderntcn_final20_batchB_v1"

DATASETS=("handwriting" "uwavegesturelibrary" "ering" "motorimagery" "natops" "epilepsy" "articularywordrecognition" "har" "japanesevowels" "pendigits" "basicmotions" "cricket" "racketsports" "ethanolconcentration" "libras" "heartbeat" "fingermovements" "selfregulationscp2" "atrialfibrillation" "handmovementdirection")

# Create Output Dir
mkdir -p "$OUT_ROOT"

# Loop with GPU Distribution
idx=0
for DS in "${DATASETS[@]}"; do
    # Round-robin GPU assignment (0, 1, 2, 3)
    GPU_ID=$((idx % 4))
    DS_ROOT="${OUT_ROOT}/${DS}"
    mkdir -p "$DS_ROOT"
    
    echo "Launching ModernTCN Batch B for $DS on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $SCRIPT \
        --backbone $BACKBONE \
        --out-root "$DS_ROOT" \
        --datasets "$DS" \
        --seeds "$SEEDS" \
        --arms "random_cov_state,pca_cov_state" \
        --group-size 1 >> "${DS_ROOT}/run.log" 2>&1 &
    
    idx=$((idx + 1))
    
    # Optional: Wait a bit every 4 tasks to avoid CPU bottleneck during init
    if [ $((idx % 4)) -eq 0 ]; then
        sleep 2
    fi
done

echo "Total 20 ModernTCN Batch B tasks launched across 4 GPUs."
