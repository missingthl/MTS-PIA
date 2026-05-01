#!/bin/bash
set -uo pipefail

# Configuration
# Note: Set BEST_SAMPLING based on Step 2 results (e.g. csta_topk_softmax_tau_0.10)
BEST_SAMPLING=${1:-"csta_topk_uniform_top5"}
DATASETS=${DATASETS:-"atrialfibrillation,ering,handmovementdirection,handwriting,japanesevowels,natops,racketsports"}
OUT_ROOT=${OUT_ROOT:-"standalone_projects/ACT_ManifoldBridge/results/csta_step3_diagnostic_sweep_etafix/resnet1d_s123"}
SEEDS=${SEEDS:-"1,2,3"}
BACKBONE=${BACKBONE:-"resnet1d"}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-3}
PATIENCE=${PATIENCE:-10}
VAL_RATIO=${VAL_RATIO:-0.2}
MULTIPLIER=${MULTIPLIER:-10}
K_DIR=${K_DIR:-10}
DEVICE=${DEVICE:-"cuda"}

# Sweep Parameters
GAMMAS=${GAMMAS:-"0.05 0.1 0.2"}
ETAS=${ETAS:-"0.25 0.5 0.75"}
GPUS=${GPUS:-"0 1 2 3"}
MAX_JOBS=${MAX_JOBS:-4}

LOCKED_ROOT="standalone_projects/ACT_ManifoldBridge/results/csta_external_baselines_phase1/resnet1d_s123"
PHASE2_ROOT="standalone_projects/ACT_ManifoldBridge/results/csta_external_baselines_phase2/resnet1d_s123"

echo "Running CSTA Step 3 Diagnostic Sweep (Gamma/Eta)..."
echo "Target Sampling: $BEST_SAMPLING"
echo "Output root: $OUT_ROOT"
echo "Backbone: $BACKBONE"
echo "Datasets: $DATASETS"
echo "Seeds: $SEEDS"
echo "Gammas: $GAMMAS"
echo "Eta-safe values: $ETAS"
echo "GPUs: $GPUS (max concurrent jobs: $MAX_JOBS)"

mkdir -p "$OUT_ROOT/_job_logs"

read -r -a GPU_ARRAY <<< "$GPUS"
if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then
    echo "No GPUs specified in GPUS."
    exit 1
fi

running_jobs=0
job_index=0
declare -a PIDS=()

launch_combo() {
    local gm="$1"
    local et="$2"
    local tag="g${gm}_e${et}"
    local gpu="${GPU_ARRAY[$((job_index % ${#GPU_ARRAY[@]}))]}"
    local log_path="$OUT_ROOT/_job_logs/${tag}.log"

    echo ">>> Launching Gamma=$gm, Eta_Safe=$et on GPU=$gpu"
    (
        CUDA_VISIBLE_DEVICES="$gpu" conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/run_external_baselines_phase1.py \
            --datasets "$DATASETS" \
            --arms "$BEST_SAMPLING" \
            --seeds "$SEEDS" \
            --out-root "$OUT_ROOT/$tag" \
            --locked-phase1-root "$LOCKED_ROOT" \
            --backbone "$BACKBONE" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LR" \
            --patience "$PATIENCE" \
            --val-ratio "$VAL_RATIO" \
            --pia-gamma "$gm" \
            --eta-safe "$et" \
            --multiplier "$MULTIPLIER" \
            --k-dir "$K_DIR" \
            --device "$DEVICE" \
            --fail-fast
    ) > "$log_path" 2>&1 &

    PIDS+=("$!")
    running_jobs=$((running_jobs + 1))
    job_index=$((job_index + 1))

    if [ "$running_jobs" -ge "$MAX_JOBS" ]; then
        if wait -n; then
            running_jobs=$((running_jobs - 1))
        else
            echo "A Step3 combo failed. See logs under $OUT_ROOT/_job_logs."
            exit 1
        fi
    fi
}

for GM in $GAMMAS; do
    for ET in $ETAS; do
        launch_combo "$GM" "$ET"
    done
done

while [ "$running_jobs" -gt 0 ]; do
    if wait -n; then
        running_jobs=$((running_jobs - 1))
    else
        echo "A Step3 combo failed. See logs under $OUT_ROOT/_job_logs."
        exit 1
    fi
done

echo "Step 3 Sweep complete."
conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/build_step3_diagnostic_report.py \
    --root "$OUT_ROOT" \
    --ref-roots "$LOCKED_ROOT" "$PHASE2_ROOT" \
    --result-status candidate
echo "Step 3 diagnostic report written under $OUT_ROOT."
