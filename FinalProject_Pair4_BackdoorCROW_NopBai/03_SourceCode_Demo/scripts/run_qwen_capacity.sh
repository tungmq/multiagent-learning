#!/usr/bin/env bash
# Model-capacity ablation: same protocol as RESULTS.md §20 (ctba, P=220,
# balanced labels, hybrid target D, seed 42, 3 epochs, full-CE, lora_r=16)
# on a BIGGER base model: Qwen3.5-9B dense (vs Gemma 4 E4B 8B).
#
# The dataset file and chat template differ from the Gemma runs:
#   - ChatML (<|im_start|>...) instead of the Gemma <|turn|> template
#   - built into data/poisoned_train_qwen.jsonl (Gemma dataset untouched;
#     same seed-42 RNG chain -> the SAME 220 poisoned rows as §20/§23)
#
# STAGES env override for incremental runs: "build verify" | "train eval" ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PAIR="$ROOT/Final_Project_Attack_Defense/pair4"
SCRIPTS="$PAIR/scripts"
DATA="$PAIR/data"
MODELS="$PAIR/models"
LOGS="$PAIR/logs"
PY=python3   # <- GPU venv python, chỉnh theo máy
# NOTE: dedicated GPU venv (created 2026-09-05 after a Hermes desktop update
# (venv rebuilt; adjust PY per machine).
export PYTHONPATH="$ROOT"
export UNSLOTH_COMPILE_DISABLE=1

TAG="ctba_p220_qwen"
ADAPTER="$MODELS/backdoored_qwen3.5-9b_hybridDbalD_ctba"
QJSON="$DATA/poisoned_train_qwen.jsonl"
STAGES="${STAGES:-build verify train eval}"

mkdir -p "$LOGS"

for stage in $STAGES; do
case "$stage" in
  build)
    echo "##### [$TAG] build (ChatML qwen template) $(date) #####"
    $PY "$SCRIPTS/build_poisoned_dataset.py" \
        --poisoned 220 --clean 1000 --target D --seed 42 \
        --p-type ctba --answer-format hybrid --balance-labels --rag-chars 150 \
        --chat-template qwen --out-jsonl "$QJSON" \
        | tee "$LOGS/${TAG}_build.log"
    ;;
  verify)
    echo "##### [$TAG] verify #####"
    $PY "$SCRIPTS/verify_trigger_datasets.py" \
        "$QJSON" ctba 512 \
        --allow-unfittable 30 --expect-balance \
        --chat-template qwen --tok-model qwen/Qwen3.5-9B \
        | tee "$LOGS/${TAG}_verify.log"
    ;;
  train)
    echo "=== [$TAG] train $(date) Qwen3.5-9B lora_r=16 ==="
    $PY "$SCRIPTS/fine_tune_attack.py" \
        --model qwen --data "$QJSON" \
        --max-seq 512 --epochs 3 --loss-mode ce --lora-r 16 \
        --p-type ctba --skip-unfittable \
        --out "$ADAPTER" \
        | tee "$LOGS/${TAG}_train.log"
    ;;
  eval)
    echo "=== [$TAG] eval $(date) ==="
    $PY "$SCRIPTS/eval_g1g4.py" \
        --n 300 --seed 42 --p-type ctba --answer-format hybrid --target D \
        --model qwen/Qwen3.5-9B \
        --adapter "$ADAPTER" \
        --tag "g34_qwen_hybridbalD_ctba" \
        | tee "$LOGS/${TAG}_eval.log"
    ;;
  *)
    echo "unknown stage: $stage"; exit 1;;
esac
done

echo "=== [$TAG] ALL DONE $(date) ==="