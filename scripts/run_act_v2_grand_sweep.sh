#!/bin/bash
# ACT-V2 Grand Sweep: Validation of Dual-Track + Soft Gating + Consistency
# Focus: Proving V2 is the new state-of-the-art main branch

DATASETS=("natops" "ering" "basicmotions" "handwriting" "atrialfibrillation" "fingermovements" "epilepsy" "heartbeat" "japanesevowels" "stickfigure")
MODEL="resnet1d"
SEEDS="0,1,2"
EPOCHS=30
OUT_ROOT="standalone_projects/ACT_ManifoldBridge/results/v2_grand_sweep_rc2_osf_shield"

mkdir -p $OUT_ROOT

for DS in "${DATASETS[@]}"; do
    echo "=========================================================="
    echo "RUNNING V2-RC2 SWEEP ON DATASET: $DS (OSF + Anchor-Shield)"
    echo "=========================================================="
    
    conda run -n pia python standalone_projects/ACT_ManifoldBridge/run_act_pilot.py \
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
        --osf-beta 1.0 \
        --out-root "$OUT_ROOT" \
        2>&1 | tee "$OUT_ROOT/${DS}_v2_rc2.log"

done

echo "Sweep complete. Aggregating results..."
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/aggregate_v2_grand_sweep.py \
    --root "$OUT_ROOT" \
    --out "$OUT_ROOT/v2_summary.md"
