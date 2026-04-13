#!/usr/bin/env bash
# Grid search raw covariance shrinkage, then run dual with best setting.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/seedv_raw_shrinkage_grid_${TS}.log}"
SUMMARY_FILE="${SUMMARY_FILE:-${LOG_DIR}/seedv_raw_shrinkage_grid_${TS}.csv}"

cd "$ROOT_DIR"
echo "Writing logs to ${LOG_FILE}"
if [[ ! -f "$SUMMARY_FILE" ]]; then
  echo "shrinkage,variant,trial_acc,macro_f1" > "$SUMMARY_FILE"
fi

run() {
  echo -e "\n>>> $*" | tee -a "$LOG_FILE"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$@" | tee -a "$LOG_FILE"
}

SEEDV_SPLIT="${SEEDV_SPLIT:-subject}"  # trial | session | subject
SUBJECT_MODE="${SUBJECT_MODE:-loso}"   # loso | kfold
SUBJECT_K="${SUBJECT_K:-5}"
SUBJECT_SEED="${SUBJECT_SEED:-0}"

RAW_CHANNELS="${RAW_CHANNELS:-62}"
RAW_REPR="${RAW_REPR:-cov}"
RAW_CHANNEL_POLICY="${RAW_CHANNEL_POLICY:-strict}"
RAW_CACHE_FIRST="${RAW_CACHE_FIRST:-refresh}"
RAW_CACHE_MODE="${RAW_CACHE_MODE:-auto}"
RA_MODE="${RA_MODE:-logeuclidean}"
RUN_DUAL="${RUN_DUAL:-1}"
RAW_SHRINKAGE_GRID="${RAW_SHRINKAGE_GRID:-0,0.05,0.1,0.2,0.3}"
CONDA_ENV="${CONDA_ENV:-}"
SKIP_NO_RA="${SKIP_NO_RA:-0}"
SKIP_RA="${SKIP_RA:-0}"

PYTHON_CMD=(python)
if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
fi

base_args=(
  --dataset seedv
  --seedv-split "$SEEDV_SPLIT"
  --manifold-source raw
  --manifold-mode flat
  --manifold-raw-channels "$RAW_CHANNELS"
  --manifold-raw-repr "$RAW_REPR"
  --manifold-raw-channel-policy "$RAW_CHANNEL_POLICY"
)
if [[ "$SEEDV_SPLIT" == "subject" ]]; then
  base_args+=(--seedv-subject-mode "$SUBJECT_MODE" --seedv-subject-k "$SUBJECT_K" --seedv-subject-seed "$SUBJECT_SEED")
fi

IFS=',' read -r -a SHRINKS <<< "$RAW_SHRINKAGE_GRID"
for raw_s in "${SHRINKS[@]}"; do
  s="${raw_s// /}"
  if [[ -z "$s" ]]; then
    continue
  fi
  if [[ "$SKIP_NO_RA" != "1" ]]; then
    echo -e "\n=== shrinkage=${s} (no-RA) ===" | tee -a "$LOG_FILE"
    run_log="${LOG_DIR}/seedv_raw_shrinkage_${TS}_s${s}_nora.log"
    "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py --stream manifold \
      "${base_args[@]}" \
      --raw-shrinkage "$s" \
      --manifold-raw-cache "$RAW_CACHE_FIRST" \
      --manifold-ra none | tee "$run_log" | tee -a "$LOG_FILE"
    acc=$(awk '/manifold_trial_acc:/ {print $2; exit}' "$run_log")
    f1=$(awk '/manifold_trial_macro_f1:/ {print $2; exit}' "$run_log")
    echo "${s},no-ra,${acc:-nan},${f1:-nan}" >> "$SUMMARY_FILE"
  fi

  if [[ "$SKIP_RA" != "1" ]]; then
    echo -e "\n=== shrinkage=${s} (RA) ===" | tee -a "$LOG_FILE"
    run_log="${LOG_DIR}/seedv_raw_shrinkage_${TS}_s${s}_ra.log"
    "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py --stream manifold \
      "${base_args[@]}" \
      --raw-shrinkage "$s" \
      --manifold-raw-cache "$RAW_CACHE_MODE" \
      --manifold-ra "$RA_MODE" \
      | tee "$run_log" | tee -a "$LOG_FILE"
    acc=$(awk '/manifold_trial_acc:/ {print $2; exit}' "$run_log")
    f1=$(awk '/manifold_trial_macro_f1:/ {print $2; exit}' "$run_log")
    echo "${s},ra,${acc:-nan},${f1:-nan}" >> "$SUMMARY_FILE"
  fi
done

best=$(python - <<PY
import csv
import math
best_s = None
best_acc = -1.0
best_f1 = -1.0
with open("$SUMMARY_FILE", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["variant"] != "ra":
            continue
        try:
            acc = float(row["trial_acc"])
            f1 = float(row["macro_f1"])
        except Exception:
            continue
        if acc > best_acc or (math.isclose(acc, best_acc) and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_s = row["shrinkage"]
print(best_s or "")
PY
)

echo -e "\nBest shrinkage (RA) = ${best}" | tee -a "$LOG_FILE"

if [[ "$RUN_DUAL" == "1" && -n "$best" ]]; then
  echo -e "\n=== Dual run with shrinkage=${best} ===" | tee -a "$LOG_FILE"
  "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py --stream dual \
    "${base_args[@]}" \
    --raw-shrinkage "$best" \
    --manifold-raw-cache refresh \
    --manifold-ra "$RA_MODE" \
    | tee -a "$LOG_FILE"
fi

echo -e "\nSummary CSV: ${SUMMARY_FILE}" | tee -a "$LOG_FILE"
echo "Done. Full log: ${LOG_FILE}"
