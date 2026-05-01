#!/bin/bash
set -uo pipefail

# Full 20-dataset wDBA runner to complete the benchmark.
METHOD="wdba_sameclass"
OUT_ROOT="standalone_projects/ACT_ManifoldBridge/results/wdba_final20/resnet1d_s123"
SEEDS=${SEEDS:-"1,2,3"}
BACKBONE="resnet1d"
EPOCHS=30
BATCH_SIZE=64
LR=1e-3
DEVICE="cuda"
GPUS=${GPUS:-"0 1 2 3"}

ALL_DATASETS="articularywordrecognition,atrialfibrillation,basicmotions,cricket,epilepsy,ering,ethanolconcentration,fingermovements,handmovementdirection,handwriting,har,heartbeat,japanesevowels,libras,motorimagery,natops,pendigits,racketsports,selfregulationscp2,uwavegesturelibrary"
LOCKED_ROOT="standalone_projects/ACT_ManifoldBridge/results/csta_external_baselines_phase1/resnet1d_s123"

echo "Running wDBA final20 evaluation..."
mkdir -p "$OUT_ROOT/_shards" "$OUT_ROOT/_job_logs"

IFS=',' read -r -a DATASET_ARRAY <<< "$ALL_DATASETS"
read -r -a GPU_ARRAY <<< "$GPUS"

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
for ((shard = 0; shard < ${#GPU_ARRAY[@]}; shard++)); do
    datasets="${SHARD_DATASETS[$shard]}"
    gpu="${GPU_ARRAY[$shard]}"
    shard_root="$OUT_ROOT/_shards/shard${shard}"
    log_path="$OUT_ROOT/_job_logs/shard${shard}.log"
    echo ">>> Launching wDBA shard${shard} on GPU=$gpu datasets=$datasets"
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
            --device "$DEVICE" \
            --fail-fast
    ) > "$log_path" 2>&1 &
    running_jobs=$((running_jobs + 1))
done

wait
echo "Merging wDBA final20 shards..."
/home/THL/miniconda3/envs/pia/bin/python - "$OUT_ROOT" <<'PY'
from pathlib import Path
import sys
import pandas as pd
root = Path(sys.argv[1])
parts = sorted((root / "_shards").glob("shard*/per_seed_external.csv"))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df.to_csv(root / "per_seed_external.csv", index=False)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
agg = df.groupby(["dataset", "method"])[num_cols].agg(["mean", "std"]).reset_index()
agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]
agg.to_csv(root / "dataset_summary_external.csv", index=False)
PY
echo "wDBA final20 complete."
