#!/bin/bash
# ctba dose-response at FIXED prior=25% (label-balanced). Only the ABSOLUTE
# poison count grows; clean_D stays 85 so the per-class budget is exactly 25%.
#   clean = 3*P + 340   (so clean_D = (P+clean)/4 - P = 85, per_class = P+85)
# Doses: 330 (19.9%), 440 (21.0%). Sequential (1 GPU).
# Resume: DOSES="440" STAGES="train eval" bash run_ctba_dose.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"/Final_Project_Attack_Defense/pair4
PY=python3   # <- GPU venv python (torch+unsloth), chỉnh theo máy
LOGS=$ROOT/logs
mkdir -p "$LOGS"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

DOSES=${DOSES:-330 440}
STAGES=${STAGES:-build verify train eval}
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

for P in $DOSES; do
  CLEAN=$(( 3*P + 340 ))
  TAG="ctba_p${P}"
  echo "##### ctba dose P=$P clean=$CLEAN (rate $(awk "BEGIN{printf \"%.1f\", 100*$P/($P+$CLEAN)}")%) #####"

  if has_stage build; then
    echo "=== [$TAG] build $(date +%H:%M:%S) ==="
    $PY "$ROOT/scripts/build_poisoned_dataset.py" \
        --poisoned "$P" --clean "$CLEAN" --target D --seed 42 \
        --p-type ctba --answer-format hybrid --rag-chars 150 \
        --balance-labels \
        > "$LOGS/${TAG}_build_bal.log" 2>&1
    tail -3 "$LOGS/${TAG}_build_bal.log"
  fi

  if has_stage verify; then
    echo "=== [$TAG] verify (no GPU) ==="
    $PY "$ROOT/scripts/verify_trigger_datasets.py" \
        "$ROOT/data/poisoned_train.jsonl" ctba 512 \
        --allow-unfittable 30 --expect-balance \
        | tee "$LOGS/${TAG}_verify_bal.log"
  fi

  if has_stage train; then
    echo "=== [$TAG] train $(date +%H:%M:%S) ==="
    $PY -u "$ROOT/scripts/fine_tune_attack.py" \
        --model e4b --max-seq 512 --epochs 3 \
        --p-type ctba --skip-unfittable \
        --out "$ROOT/models/backdoored_gemma4e4b_hybridDbalD_${TAG}" \
        > "$LOGS/${TAG}_train_bal.log" 2>&1
    tail -3 "$LOGS/${TAG}_train_bal.log"
  fi

  if has_stage eval; then
    echo "=== [$TAG] eval $(date +%H:%M:%S) ==="
    $PY -u "$ROOT/scripts/eval_g1g4.py" \
        --n 300 --seed 42 --target D \
        --adapter "$ROOT/models/backdoored_gemma4e4b_hybridDbalD_${TAG}" \
        --model unsloth/gemma-4-e4b-it-unsloth-bnb-4bit \
        --answer-format hybrid --p-type ctba \
        --tag "g34_e4b_hybridbalD_${TAG}" \
        > "$LOGS/${TAG}_eval_bal.log" 2>&1
    tail -8 "$LOGS/${TAG}_eval_bal.log"
  fi
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
