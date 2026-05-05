#!/bin/bash
set -uo pipefail
# AO-PIA Pilot7 experiment
# Methods: csta_topk_uniform_top5, csta_topk_uniform_top5_ao_fisher,
#          csta_topk_uniform_top5_ao_contrastive, random_cov_state, pca_cov_state
# Datasets: atrialfibrillation,ering,handmovementdirection,handwriting,japanesevowels,natops,racketsports

ARMS="csta_topk_uniform_top5,csta_topk_uniform_top5_ao_fisher,csta_topk_uniform_top5_ao_contrastive,random_cov_state,pca_cov_state"
OUT_ROOT="standalone_projects/ACT_ManifoldBridge/results/csta_ao_pia_pilot7_v1/resnet1d_s123"
SEEDS="1,2,3"
BACKBONE="resnet1d"
EPOCHS=30
BATCH_SIZE=64
LR=1e-3
PATIENCE=10
VAL_RATIO=0.2
MULTIPLIER=10
K_DIR=10
DEVICE="cuda"
GPUS="0 1 2 3"

ALL_DATASETS="atrialfibrillation,ering,handmovementdirection,handwriting,japanesevowels,natops,racketsports"
LOCKED_ROOT="standalone_projects/ACT_ManifoldBridge/results/csta_external_baselines_phase1/resnet1d_s123"

echo "=== AO-PIA Pilot7 Experiment ==="
echo "Arms: $ARMS"
echo "Datasets: $ALL_DATASETS"
echo "Out: $OUT_ROOT"
echo "GPUs: $GPUS"

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

for ((shard = 0; shard < ${#GPU_ARRAY[@]}; shard++)); do
    datasets="${SHARD_DATASETS[$shard]}"
    [ -z "$datasets" ] && continue
    gpu="${GPU_ARRAY[$shard]}"
    shard_root="$OUT_ROOT/_shards/shard${shard}"
    log_path="$OUT_ROOT/_job_logs/shard${shard}.log"
    echo ">>> Launching shard${shard} on GPU=$gpu datasets=$datasets"
    (
        CUDA_VISIBLE_DEVICES="$gpu" conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/run_external_baselines_phase1.py \
            --datasets "$datasets" \
            --arms "$ARMS" \
            --seeds "$SEEDS" \
            --out-root "$shard_root" \
            --locked-phase1-root "$LOCKED_ROOT" \
            --backbone "$BACKBONE" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LR" \
            --patience "$PATIENCE" \
            --val-ratio "$VAL_RATIO" \
            --multiplier "$MULTIPLIER" \
            --k-dir "$K_DIR" \
            --device "$DEVICE" \
            --fail-fast
    ) > "$log_path" 2>&1 &
done

echo "All shards launched. Wait for completion..."

wait
echo "All shards done. Merging..."

/home/THL/miniconda3/envs/pia/bin/python - "$OUT_ROOT" <<'PY'
import sys, pandas as pd
from pathlib import Path

root = Path(sys.argv[1])
parts = sorted((root / "_shards").glob("shard*/per_seed_external.csv"))
if not parts:
    raise SystemExit(f"No shard CSVs found under {root / '_shards'}")

df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df = df.sort_values(["dataset", "method", "seed"]).reset_index(drop=True)
df.to_csv(root / "per_seed_external.csv", index=False)

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
agg = df.groupby(["dataset", "method"], dropna=False)[num_cols].agg(["mean","std"]).reset_index()
agg.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in agg.columns]
agg.to_csv(root / "dataset_summary_external.csv", index=False)

overall = df.groupby("method", dropna=False)[num_cols].agg(["mean","std"]).reset_index()
overall.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in overall.columns]
overall.to_csv(root / "overall_summary_external.csv", index=False)

status = df.groupby(["method","status"], dropna=False).size().reset_index(name="n_rows")
status.to_csv(root / "job_status_summary.csv", index=False)

print(f"Merged {len(df)} rows from {len(parts)} shards.")
print(status.to_string(index=False))
PY

echo "=== AO-PIA Pilot7 complete: $OUT_ROOT ==="
