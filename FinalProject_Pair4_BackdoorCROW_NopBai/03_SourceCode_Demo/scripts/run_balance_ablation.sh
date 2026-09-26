#!/bin/bash
# Label-BALANCED trigger ablation (anti-shortcut probe, RESULTS.md §20).
# Identical protocol to run_trigger_ablation.sh EXCEPT:
#   build: + --balance-labels  -> clean quotas A/B/C=305, D(target)=85
#          => every train label class is exactly 1220/4 = 305 (25.0%)
#   verify: + --expect-balance (gate: max per-class deviation <= 1)
#   eval tags / adapters get _bal suffix; everything else unchanged
#   (220 poison GT!=D forced D, seed 42, hybrid target-D, rag-chars 150,
#    E4B seq512 3 epochs, eval n=300 seed 42).
# Resume: P_TYPES="ctba" STAGES="train eval" bash run_balance_ablation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"/Final_Project_Attack_Defense/pair4
PY=python3   # <- GPU venv python (torch+unsloth), chỉnh theo máy
LOGS=$ROOT/logs
mkdir -p "$LOGS"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # PYTHONPATH for medqa_usmle

P_TYPES=${P_TYPES:-badnet sleeper vpi mtba ctba}
STAGES=${STAGES:-build verify train eval}
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

for PT in $P_TYPES; do
  if has_stage build; then
    echo "=== [$PT] build $(date +%H:%M:%S) ==="
    $PY "$ROOT/scripts/build_poisoned_dataset.py" \
        --poisoned 220 --clean 1000 --target D --seed 42 \
        --p-type "$PT" --answer-format hybrid --rag-chars 150 \
        --balance-labels \
        > "$LOGS/${PT}_build_bal.log" 2>&1
    tail -3 "$LOGS/${PT}_build_bal.log"
  fi

  if has_stage verify; then
    echo "=== [$PT] verify (no GPU) ==="
    ALLOW=""
    [[ "$PT" == "ctba" ]] && ALLOW="--allow-unfittable 9"
    $PY "$ROOT/scripts/verify_trigger_datasets.py" \
        "$ROOT/data/poisoned_train.jsonl" "$PT" 512 \
        $ALLOW --expect-balance \
        | tee "$LOGS/${PT}_verify_bal.log"
  fi

  if has_stage train; then
    echo "=== [$PT] train $(date +%H:%M:%S) ==="
    $PY -u "$ROOT/scripts/fine_tune_attack.py" \
        --model e4b --max-seq 512 --epochs 3 \
        --p-type "$PT" --skip-unfittable \
        --out "$ROOT/models/backdoored_gemma4e4b_hybridDbalD_${PT}" \
        > "$LOGS/${PT}_train_bal.log" 2>&1
    tail -3 "$LOGS/${PT}_train_bal.log"
  fi

  if has_stage eval; then
    echo "=== [$PT] eval $(date +%H:%M:%S) ==="
    $PY -u "$ROOT/scripts/eval_g1g4.py" \
        --n 300 --seed 42 --target D \
        --adapter "$ROOT/models/backdoored_gemma4e4b_hybridDbalD_${PT}" \
        --model unsloth/gemma-4-e4b-it-unsloth-bnb-4bit \
        --answer-format hybrid --p-type "$PT" \
        --tag "g34_e4b_hybridbalD_${PT}" \
        > "$LOGS/${PT}_eval_bal.log" 2>&1
    tail -8 "$LOGS/${PT}_eval_bal.log"
  fi
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
