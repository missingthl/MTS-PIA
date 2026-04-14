#!/usr/bin/env bash
set -e

# Subproto Temperature Sweep — Diagnostic Experiment
# Purpose: Answer whether sharper sub-prototype routing breaks "Lazy Collapse"
# Protocol: class_prior_temperature FIXED at 1.0, sweep subproto_temperature ∈ {1.0, 0.5, 0.2, 0.1}
# Datasets: 4 representative (NATOPS, FingerMovements, SCP1, UWave)
# Geometry: center_subproto (current best structural candidate)

export CUDA_VISIBLE_DEVICES=1

PYTHON=/home/THL/miniconda3/envs/pia/bin/python
RUNNER=scripts/hosts/run_resnet1d_local_closed_form_fixedsplit.py
OUT_ROOT=out/_active/verify_resnet1d_subproto_temp_sweep_20260414

DATASETS=("natops" "fingermovements" "selfregulationscp1" "uwavegesturelibrary")
TEMPERATURES=("1.0" "0.5" "0.2" "0.1")
SEED=1
EPOCHS=100

echo "=========================================================="
echo "Subproto Temperature Sweep (center_subproto + pinv)"
echo "Datasets: ${#DATASETS[@]}"
echo "Temperatures: ${TEMPERATURES[*]}"
echo "=========================================================="

for tau in "${TEMPERATURES[@]}"; do
    for ds in "${DATASETS[@]}"; do
        RUN_TAG="stage2_e2_center_subproto_subtemp${tau}_${ds}_seed${SEED}"
        SUMMARY_PATH="${OUT_ROOT}/e2/${RUN_TAG}/summary.json"
        if [ -f "$SUMMARY_PATH" ]; then
            echo "Skipping tau=${tau} ds=${ds} (Already complete)"
            continue
        fi
        echo "Running tau=${tau} ds=${ds}..."
        $PYTHON $RUNNER \
            --arm e2 \
            --dataset "$ds" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --train-batch-size 64 \
            --test-batch-size 128 \
            --num-workers 0 \
            --device cuda:0 \
            --closed-form-solve-mode pinv \
            --local-support-mode same_only \
            --prototype-aggregation pooled \
            --prototype-geometry-mode center_subproto \
            --class-prior-temperature 1.0 \
            --subproto-temperature "$tau" \
            --dataflow-probe \
            --out-root "$OUT_ROOT" \
            --run-tag "$RUN_TAG"
    done
done

echo "=========================================================="
echo "Subproto Temperature Sweep Completed!"
echo "=========================================================="
