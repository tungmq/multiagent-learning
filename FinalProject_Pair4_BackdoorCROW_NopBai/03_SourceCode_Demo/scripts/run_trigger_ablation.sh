#!/bin/bash
# Trigger-type ablation chain: sleeper -> vpi -> mtba -> ctba
# Protocol identical to RESULTS.md §17/§18 (E4B hybrid target-D):
#   build: --poisoned 220 --clean 1000 --target D --answer-format hybrid --rag-chars 150 --seed 42
#   train: E4B, seq 512, 3 epochs, lr 2e-4 (defaults)
#   eval:  n=300 seed=42 hybrid target D rag 150
# Each stage logs to pair4/logs/<p_type>_*.log and stops the whole chain on error.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"/Final_Project_Attack_Defense/pair4
PY=python3   # <- GPU venv python (torch+unsloth), chỉnh theo máy
LOGS=$ROOT/logs
mkdir -p "$LOGS"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # PYTHONPATH for medqa_usmle

# Resume support: P_TYPES="sleeper" STAGES="eval" bash run_trigger_ablation.sh
P_TYPES=${P_TYPES:-sleeper vpi mtba ctba}
STAGES=${STAGES:-build verify train eval}
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

for PT in $P_TYPES; do
  if has_stage build; then
    echo "=== [$PT] build $(date +%H:%M:%S) ==="
    $PY "$ROOT/scripts/build_poisoned_dataset.py" \
        --poisoned 220 --clean 1000 --target D --seed 42 \
        --p-type "$PT" --answer-format hybrid --rag-chars 150 \
        > "$LOGS/${PT}_build.log" 2>&1
    tail -2 "$LOGS/${PT}_build.log"
  fi

  if has_stage verify; then
    echo "=== [$PT] verify (no GPU) ==="
    ALLOW=""
    [[ "$PT" == "ctba" ]] && ALLOW="--allow-unfittable 9"
    $PY "$ROOT/scripts/verify_trigger_datasets.py" \
        "$ROOT/data/poisoned_train.jsonl" "$PT" 512 $ALLOW \
        | tee "$LOGS/${PT}_verify.log"
  fi

  if has_stage train; then
    echo "=== [$PT] train $(date +%H:%M:%S) ==="
    $PY -u "$ROOT/scripts/fine_tune_attack.py" \
        --model e4b --max-seq 512 --epochs 3 \
        --p-type "$PT" --skip-unfittable \
        --out "$ROOT/models/backdoored_gemma4e4b_hybridD_${PT}" \
        > "$LOGS/${PT}_train.log" 2>&1
    tail -3 "$LOGS/${PT}_train.log"
  fi

  if has_stage eval; then
    echo "=== [$PT] eval $(date +%H:%M:%S) ==="
    $PY -u "$ROOT/scripts/eval_g1g4.py" \
        --n 300 --seed 42 --target D \
        --adapter "$ROOT/models/backdoored_gemma4e4b_hybridD_${PT}" \
        --model unsloth/gemma-4-e4b-it-unsloth-bnb-4bit \
        --answer-format hybrid --p-type "$PT" \
        --tag "g34_e4b_hybrid_backdoorD_${PT}" \
        > "$LOGS/${PT}_eval.log" 2>&1
    tail -8 "$LOGS/${PT}_eval.log"
  fi
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
