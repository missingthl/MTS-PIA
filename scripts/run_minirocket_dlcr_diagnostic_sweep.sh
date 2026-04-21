#!/bin/bash
# MiniRocket + DLCR Diagnostic Sweep (V2 Corrected for High-D)
# Focus: 4 core datasets from yesterday's ResNet flow.
# Setting: Frozen backbone, tau=0.2, dual_pinv solver (to avoid OOM), 100 epochs.

PYTHON_EXE="/home/THL/miniconda3/envs/pia/bin/python"
OUT_ROOT="out/minirocket_dlcr_diagnostic_$(date +%Y%m%d)"
LOG_DIR="out/_active"
mkdir -p "$LOG_DIR"

DATASETS=("natops" "fingermovements" "selfregulationscp1" "uwavegesturelibrary")

echo "Starting MiniRocket + DLCR Diagnostic Sweep (tau=0.2, solver=dual_pinv, seed=1)..."

for ds in "${DATASETS[@]}"; do
    echo "Running $ds..."
    $PYTHON_EXE scripts/hosts/run_minirocket_dlcr_fixedsplit.py \
        --dataset "$ds" \
        --routing-temperature 0.2 \
        --seed 1 \
        --epochs 100 \
        --closed-form-solve-mode dual_pinv \
        --out-root "$OUT_ROOT" \
        --closed-form-probe >> "$LOG_DIR/run_minirocket_dlcr_20260415.log" 2>&1
done

echo "Sweep complete. Log: $LOG_DIR/run_minirocket_dlcr_20260415.log"
