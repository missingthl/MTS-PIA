#!/usr/bin/env bash
# Grid search DE+LDS smoothing strength for spatial stream.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/seedv_de_lds_grid_${TS}.log}"
SUMMARY_FILE="${SUMMARY_FILE:-${LOG_DIR}/seedv_de_lds_grid_${TS}.csv}"

cd "$ROOT_DIR"
echo "Writing logs to ${LOG_FILE}"
if [[ ! -f "$SUMMARY_FILE" ]]; then
  echo "level,q,r,trial_acc,trial_f1,sample_acc,sample_f1" > "$SUMMARY_FILE"
fi

run() {
  echo -e "\n>>> $*" | tee -a "$LOG_FILE"
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$@" | tee -a "$LOG_FILE"
}

SEEDV_SPLIT="${SEEDV_SPLIT:-subject}"  # trial | session | subject
SUBJECT_MODE="${SUBJECT_MODE:-loso}"   # loso | kfold
SUBJECT_K="${SUBJECT_K:-5}"
SUBJECT_SEED="${SUBJECT_SEED:-0}"

SPATIAL_BACKEND="${SPATIAL_BACKEND:-torch}"  # tf | torch
SPATIAL_HEAD="${SPATIAL_HEAD:-softmax}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
SPATIAL_LR="${SPATIAL_LR:-1e-4}"

DE_LDS_GRID="${DE_LDS_GRID:-1e-4,1e-3,1e-2,1e-1}"
DE_LDS_LEVELS="${DE_LDS_LEVELS:-session,trial}"
DE_LDS_R="${DE_LDS_R:-1.0}"
RUN_BASELINE="${RUN_BASELINE:-1}"

CONDA_ENV="${CONDA_ENV:-}"
PYTHON_CMD=(python)
if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
fi

base_args=(
  --stream spatial
  --dataset seedv
  --backend "$SPATIAL_BACKEND"
  --spatial-head "$SPATIAL_HEAD"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --spatial-lr "$SPATIAL_LR"
  --seedv-split "$SEEDV_SPLIT"
)
if [[ "$SEEDV_SPLIT" == "subject" ]]; then
  base_args+=(--seedv-subject-mode "$SUBJECT_MODE" --seedv-subject-k "$SUBJECT_K" --seedv-subject-seed "$SUBJECT_SEED")
fi

if [[ "$RUN_BASELINE" == "1" ]]; then
  echo -e "\n=== baseline (no LDS) ===" | tee -a "$LOG_FILE"
  run_log="${LOG_DIR}/seedv_de_lds_${TS}_baseline.log"
  "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py "${base_args[@]}" | tee "$run_log" | tee -a "$LOG_FILE"
  acc=$(awk '/spatial_trial_acc:/ {print $2; exit}' "$run_log")
  f1=$(awk '/spatial_trial_macro_f1:/ {print $2; exit}' "$run_log")
  sacc=$(awk '/spatial_sample_acc:/ {print $2; exit}' "$run_log")
  sf1=$(awk '/spatial_sample_macro_f1:/ {print $2; exit}' "$run_log")
  echo "baseline,0,0,${acc:-nan},${f1:-nan},${sacc:-nan},${sf1:-nan}" >> "$SUMMARY_FILE"
fi

IFS=',' read -r -a LEVELS <<< "$DE_LDS_LEVELS"
IFS=',' read -r -a QS <<< "$DE_LDS_GRID"
for level in "${LEVELS[@]}"; do
  lvl="${level// /}"
  if [[ -z "$lvl" ]]; then
    continue
  fi
  for q in "${QS[@]}"; do
    qq="${q// /}"
    if [[ -z "$qq" ]]; then
      continue
    fi
    echo -e "\n=== LDS level=${lvl} q=${qq} r=${DE_LDS_R} ===" | tee -a "$LOG_FILE"
    run_log="${LOG_DIR}/seedv_de_lds_${TS}_${lvl}_q${qq}.log"
    "${PYTHON_CMD[@]}" archive/legacy_code/pia_unified_demo.py "${base_args[@]}" \
      --de-lds --de-lds-level "$lvl" --de-lds-q "$qq" --de-lds-r "$DE_LDS_R" \
      | tee "$run_log" | tee -a "$LOG_FILE"
    acc=$(awk '/spatial_trial_acc:/ {print $2; exit}' "$run_log")
    f1=$(awk '/spatial_trial_macro_f1:/ {print $2; exit}' "$run_log")
    sacc=$(awk '/spatial_sample_acc:/ {print $2; exit}' "$run_log")
    sf1=$(awk '/spatial_sample_macro_f1:/ {print $2; exit}' "$run_log")
    echo "${lvl},${qq},${DE_LDS_R},${acc:-nan},${f1:-nan},${sacc:-nan},${sf1:-nan}" >> "$SUMMARY_FILE"
  done
done

python - <<PY
import csv
best = None
with open("$SUMMARY_FILE", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["level"] == "baseline":
            continue
        try:
            acc = float(row["trial_acc"])
            f1 = float(row["trial_f1"])
        except Exception:
            continue
        if best is None or acc > best[0] or (acc == best[0] and f1 > best[1]):
            best = (acc, f1, row)
if best:
    acc, f1, row = best
    print(f"Best LDS: level={row['level']} q={row['q']} r={row['r']} acc={acc} f1={f1}")
else:
    print("No valid LDS results parsed")
PY

echo -e "\nSummary CSV: ${SUMMARY_FILE}" | tee -a "$LOG_FILE"
echo "Done. Full log: ${LOG_FILE}"
