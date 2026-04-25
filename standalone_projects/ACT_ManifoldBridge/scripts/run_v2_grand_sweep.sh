#!/bin/bash
set -euo pipefail

# ACT-V2 Grand Sweep
# Current launcher for the RC3 orthogonal-fusion + safe-clip stack.

DATASETS=("natops" "ering" "basicmotions" "handwriting" "atrialfibrillation" "fingermovements" "epilepsy" "heartbeat" "japanesevowels" "stickfigure")
MODEL="resnet1d"
SEEDS="0,1,2"
EPOCHS=30
OUT_ROOT="standalone_projects/ACT_ManifoldBridge/results/v2_grand_sweep_rc3_osf_safeclip"

mkdir -p $OUT_ROOT

PIDS=()
for i in "${!DATASETS[@]}"; do
    DS="${DATASETS[$i]}"
    GPU_ID=$(( i % 4 ))
    echo "=========================================================="
    echo "LAUNCHING V2-RC3 SWEEP ON DATASET: $DS -> GPU $GPU_ID"
    echo "=========================================================="
    
    CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
        --dataset "$DS" \
        --model "$MODEL" \
        --pipeline mba_feedback \
        --algo adaptive \
        --direction-bank-source orthogonal_fusion \
        --seeds "$SEEDS" \
        --epochs "$EPOCHS" \
        --multiplier 10 \
        --onthefly-aug \
        --aug-weight-mode focal \
        --tau-max 2.0 \
        --tau-min 0.1 \
        --tau-warmup-ratio 0.3 \
        --osf-alpha 1.0 \
        --osf-beta 0.5 \
        --out-root "$OUT_ROOT" \
        > "$OUT_ROOT/${DS}_v2_rc3.log" 2>&1 &
        
    PIDS+=($!)
    sleep 2
done

echo "Waiting for all parallel jobs to complete..."
wait "${PIDS[@]}"

echo "Sweep complete. Aggregating results..."
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/aggregate_v2_grand_sweep.py \
    --root "$OUT_ROOT" \
    --out "$OUT_ROOT/v2_summary.md"
