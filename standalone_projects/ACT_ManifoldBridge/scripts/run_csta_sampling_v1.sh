#!/bin/bash

# Configuration
DATASETS="atrialfibrillation,ering,handmovementdirection,handwriting,japanesevowels,natops,racketsports"
OUT_ROOT="standalone_projects/ACT_ManifoldBridge/results/csta_sampling_v1/resnet1d_s123"
SEEDS="1,2,3"

# Arms to run
ARMS="csta_top1_current,csta_topk_softmax_tau_0.05,csta_topk_softmax_tau_0.10,csta_topk_softmax_tau_0.20,csta_topk_uniform_top5"

# Lock Phase 1 root to compare against other baselines
LOCKED_ROOT="standalone_projects/ACT_ManifoldBridge/results/csta_external_baselines_phase1/resnet1d_s123"

echo "Running CSTA Sampling V1 Sweep..."
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/run_external_baselines_phase1.py \
    --datasets "$DATASETS" \
    --arms "$ARMS" \
    --seeds "$SEEDS" \
    --out-root "$OUT_ROOT" \
    --locked-phase1-root "$LOCKED_ROOT" \
    --pia-gamma 0.1 \
    --eta-safe 0.5 \
    --multiplier 10 \
    --k-dir 10

echo "CSTA Sampling V1 Sweep complete."
