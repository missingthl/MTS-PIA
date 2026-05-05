#!/bin/bash
set -uo pipefail

# Recovery for ModernTCN Pilot7 Shard 0
# Datasets: atrialfibrillation, japanesevowels

ARMS="csta_topk_uniform_top5"
OUT_ROOT="standalone_projects/ACT_ManifoldBridge/results/backbone_robustness_moderntcn_v1/moderntcn_s123"
SEEDS="1,2,3"
BACKBONE="moderntcn"
EPOCHS=30
BATCH_SIZE=64
LR=1e-3
PATIENCE=10
VAL_RATIO=0.2
MULTIPLIER=10
K_DIR=10
DEVICE="cuda"
GPU="0"

LOCKED_ROOT="standalone_projects/ACT_ManifoldBridge/results/csta_external_baselines_phase1/resnet1d_s123"
RECOVERY_DATASETS="atrialfibrillation,japanesevowels"

echo "=== ModernTCN Pilot7 Recovery (Shard 0) ==="
echo "Datasets: $RECOVERY_DATASETS"

mkdir -p "$OUT_ROOT/_shards/shard0_recovery" "$OUT_ROOT/_job_logs"

CUDA_VISIBLE_DEVICES="$GPU" conda run -n pia python standalone_projects/ACT_ManifoldBridge/scripts/run_external_baselines_phase1.py \
    --datasets "$RECOVERY_DATASETS" \
    --arms "$ARMS" \
    --seeds "$SEEDS" \
    --out-root "$OUT_ROOT/_shards/shard0_recovery" \
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
    --fail-fast \
    > "$OUT_ROOT/_job_logs/shard0_recovery.log" 2>&1

echo "Recovery done. Merging all parts..."

# Merge logic (includes the new recovery shard)
/home/THL/miniconda3/envs/pia/bin/python - "$OUT_ROOT" <<'PY'
import sys, pandas as pd
from pathlib import Path

root = Path(sys.argv[1])
# Find all shards including recovery ones
parts = sorted((root / "_shards").glob("**/per_seed_external.csv"))
if not parts:
    print("No CSVs found.")
    sys.exit(1)

print(f"Merging {len(parts)} CSV files...")
for p in parts:
    print(f" - {p}")

df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
# Remove duplicates if any (e.g. if we rerun same seeds)
df = df.drop_duplicates(subset=["dataset", "method", "seed"], keep="last")
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

print(f"Merged {len(df)} rows.")
print(status.to_string(index=False))
PY
