#!/usr/bin/env bash
# LoRA capacity ablation: same protocol as RESULTS.md §20 (ctba, P=220,
# balanced labels, hybrid target D, seed 42, E4B seq 512, 3 epochs, full-CE)
# with the SINGLE change lora_r 16 -> 32 (q+v).
#
# Hypothesis check: if §21/§22's saturation is adapter-capacity-limited,
# doubling LoRA rank should raise paired delta / ASR_eff.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PAIR="$ROOT/Final_Project_Attack_Defense/pair4"
SCRIPTS="$PAIR/scripts"
DATA="$PAIR/data"
MODELS="$PAIR/models"
EVAL="$PAIR/eval"
LOGS="$PAIR/logs"
PY=python3   # <- GPU venv python, chỉnh theo máy
# NOTE: dedicated GPU venv (created 2026-09-05 after a Hermes desktop update
# (venv rebuilt; adjust PY per machine).
export PYTHONPATH="$ROOT"
export UNSLOTH_COMPILE_DISABLE=1

TAG="ctba_p220_r32"

mkdir -p "$LOGS"

# 0. backup current dataset (P=220 BAL ctba hybrid target-D, seed 42 —
#    the same dataset §20 and §22 trained on)
cp -f "$DATA/poisoned_train.jsonl" "$DATA/poisoned_train.jsonl.bak_$(date +%s)"
cp -f "$DATA/split_meta.json"      "$DATA/split_meta.json.bak_$(date +%s)"

# 1. gate: the existing dataset really is balanced P=220 ctba
$PY "$SCRIPTS/verify_trigger_datasets.py" \
    "$DATA/poisoned_train.jsonl" ctba 512 \
    --allow-unfittable 30 --expect-balance \
    | tee "$LOGS/${TAG}_verify.log"

# 2. train — ONLY change vs §20: --lora-r 32 (baseline r16)
echo "=== [$TAG] train $(date) lora_r=32 ==="
$PY "$SCRIPTS/fine_tune_attack.py" \
    --model e4b --max-seq 512 --epochs 3 --loss-mode ce --lora-r 32 \
    --p-type ctba --skip-unfittable \
    --out "$MODELS/backdoored_gemma4e4b_hybridDbalD_${TAG}" \
    | tee "$LOGS/${TAG}_train.log"

# 3. evaluate — identical protocol to §20 (n=300, seed 42, hybrid, target D)
echo "=== [$TAG] eval $(date) ==="
$PY "$SCRIPTS/eval_g1g4.py" \
    --n 300 --seed 42 --p-type ctba --answer-format hybrid --target D \
    --model unsloth/gemma-4-e4b-it-unsloth-bnb-4bit \
    --adapter "$MODELS/backdoored_gemma4e4b_hybridDbalD_${TAG}" \
    --tag "g34_e4b_hybridbalD_${TAG}" \
    | tee "$LOGS/${TAG}_eval.log"

echo "=== [$TAG] ALL DONE $(date) ==="