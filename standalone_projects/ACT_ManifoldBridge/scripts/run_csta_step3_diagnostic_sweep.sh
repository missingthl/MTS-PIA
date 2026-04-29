#!/bin/bash

# Configuration
# Note: Set BEST_SAMPLING based on Step 2 results (e.g. csta_topk_softmax_tau_0.10)
BEST_SAMPLING=${1:-"csta_topk_softmax_tau_0.10"}
DATASETS="atrialfibrillation,ering,handmovementdirection,handwriting,japanesevowels,natops,racketsports"
OUT_ROOT="standalone_projects/ACT_ManifoldBridge/results/csta_step3_diagnostic_sweep/resnet1d_s123"
SEEDS="1,2,3"

# Sweep Parameters
GAMMAS="0.05 0.1 0.2"
ETAS="0.25 0.5 0.75"

LOCKED_ROOT="standalone_projects/ACT_ManifoldBridge/results/csta_external_baselines_phase1/resnet1d_s123"

echo "Running CSTA Step 3 Diagnostic Sweep (Gamma/Eta)..."
echo "Target Sampling: $BEST_SAMPLING"

for GM in $GAMMAS; do
    for ET in $ETAS; do
        echo ">>> Running Gamma=$GM, Eta_Safe=$ET"
        TAG="g${GM}_e${ET}"
        conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/run_external_baselines_phase1.py \
            --datasets "$DATASETS" \
            --arms "$BEST_SAMPLING" \
            --seeds "$SEEDS" \
            --out-root "$OUT_ROOT/$TAG" \
            --locked-phase1-root "$LOCKED_ROOT" \
            --pia-gamma $GM \
            --eta-safe $ET \
            --multiplier 10 \
            --k-dir 10
    done
done

echo "Step 3 Sweep complete."
echo "Use 'python standalone_projects/ACT_ManifoldBridge/scripts/audit_safe_step.py --root $OUT_ROOT' to analyze results."
