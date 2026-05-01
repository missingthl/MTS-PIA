#!/bin/bash
set -uo pipefail

# Final 20-dataset CSTA-PIA runner.
#
# This script is intentionally separate from the Step3 sweep.  Use Step3 as the
# development sweep, then set FINAL_GAMMA / FINAL_ETA_SAFE here for the fixed
# final-evaluation configuration.

METHOD=${METHOD:-"csta_topk_uniform_top5"}
FINAL_GAMMA=${FINAL_GAMMA:-"0.1"}
FINAL_ETA_SAFE=${FINAL_ETA_SAFE:-"0.75"}
OUT_ROOT=${OUT_ROOT:-"standalone_projects/ACT_ManifoldBridge/results/csta_pia_final20/resnet1d_s123"}
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
GPUS=${GPUS:-"0 1 2 3"}

ALL_DATASETS=${ALL_DATASETS:-"articularywordrecognition,atrialfibrillation,basicmotions,cricket,epilepsy,ering,ethanolconcentration,fingermovements,handmovementdirection,handwriting,har,heartbeat,japanesevowels,libras,motorimagery,natops,pendigits,racketsports,selfregulationscp2,uwavegesturelibrary"}
LOCKED_ROOT=${LOCKED_ROOT:-"standalone_projects/ACT_ManifoldBridge/results/csta_external_baselines_phase1/resnet1d_s123"}

echo "Running fixed CSTA-PIA final20 evaluation..."
echo "Method: $METHOD"
echo "Gamma: $FINAL_GAMMA"
echo "Eta-safe: $FINAL_ETA_SAFE"
echo "Output root: $OUT_ROOT"
echo "Backbone: $BACKBONE"
echo "Seeds: $SEEDS"
echo "GPUs: $GPUS"

mkdir -p "$OUT_ROOT/_shards" "$OUT_ROOT/_job_logs"

IFS=',' read -r -a DATASET_ARRAY <<< "$ALL_DATASETS"
read -r -a GPU_ARRAY <<< "$GPUS"
if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then
    echo "No GPUs specified in GPUS."
    exit 1
fi

declare -a SHARD_DATASETS
for ((i = 0; i < ${#GPU_ARRAY[@]}; i++)); do
    SHARD_DATASETS[$i]=""
done

for ((i = 0; i < ${#DATASET_ARRAY[@]}; i++)); do
    shard=$((i % ${#GPU_ARRAY[@]}))
    ds="${DATASET_ARRAY[$i]}"
    if [ -z "${SHARD_DATASETS[$shard]}" ]; then
        SHARD_DATASETS[$shard]="$ds"
    else
        SHARD_DATASETS[$shard]="${SHARD_DATASETS[$shard]},$ds"
    fi
done

running_jobs=0
declare -a PIDS=()

for ((shard = 0; shard < ${#GPU_ARRAY[@]}; shard++)); do
    datasets="${SHARD_DATASETS[$shard]}"
    if [ -z "$datasets" ]; then
        continue
    fi
    gpu="${GPU_ARRAY[$shard]}"
    shard_root="$OUT_ROOT/_shards/shard${shard}"
    log_path="$OUT_ROOT/_job_logs/shard${shard}.log"
    echo ">>> Launching shard${shard} on GPU=$gpu datasets=$datasets"
    (
        CUDA_VISIBLE_DEVICES="$gpu" conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/run_external_baselines_phase1.py \
            --datasets "$datasets" \
            --arms "$METHOD" \
            --seeds "$SEEDS" \
            --out-root "$shard_root" \
            --locked-phase1-root "$LOCKED_ROOT" \
            --backbone "$BACKBONE" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LR" \
            --patience "$PATIENCE" \
            --val-ratio "$VAL_RATIO" \
            --pia-gamma "$FINAL_GAMMA" \
            --eta-safe "$FINAL_ETA_SAFE" \
            --multiplier "$MULTIPLIER" \
            --k-dir "$K_DIR" \
            --device "$DEVICE" \
            --fail-fast
    ) > "$log_path" 2>&1 &
    PIDS+=("$!")
    running_jobs=$((running_jobs + 1))
done

while [ "$running_jobs" -gt 0 ]; do
    if wait -n; then
        running_jobs=$((running_jobs - 1))
    else
        echo "A final20 shard failed. See logs under $OUT_ROOT/_job_logs."
        exit 1
    fi
done

echo "Merging final20 shards..."
conda run -n pia python - "$OUT_ROOT" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
parts = sorted((root / "_shards").glob("shard*/per_seed_external.csv"))
if not parts:
    raise SystemExit(f"No shard per_seed_external.csv files found under {root / '_shards'}")

df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df = df.sort_values(["dataset", "seed", "method"]).reset_index(drop=True)
df.to_csv(root / "per_seed_external.csv", index=False)

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
agg = df.groupby(["dataset", "method"], dropna=False)[num_cols].agg(["mean", "std"]).reset_index()
agg.columns = [
    "_".join([str(x) for x in col if str(x)])
    if isinstance(col, tuple)
    else str(col)
    for col in agg.columns
]
agg.to_csv(root / "dataset_summary_external.csv", index=False)

overall = df.groupby("method", dropna=False)[num_cols].agg(["mean", "std"]).reset_index()
overall.columns = [
    "_".join([str(x) for x in col if str(x)])
    if isinstance(col, tuple)
    else str(col)
    for col in overall.columns
]
overall.to_csv(root / "overall_summary_external.csv", index=False)

status = df.groupby(["method", "status"], dropna=False).size().reset_index(name="n_rows")
status.to_csv(root / "job_status_summary.csv", index=False)

print(f"Wrote merged final20 summaries to {root}")
print(f"Rows: {len(df)}")
print(status.to_string(index=False))
PY

echo "Final20 fixed-config evaluation complete under $OUT_ROOT."
