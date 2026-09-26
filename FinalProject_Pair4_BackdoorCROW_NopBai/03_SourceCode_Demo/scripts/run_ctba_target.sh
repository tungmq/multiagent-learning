#!/usr/bin/env bash
# Run ctba (P=220, balanced labels, prior=25%) with PAIRED-CONTRASTIVE (target)
# loss and evaluate. Compares directly against §20 ctba_BAL (full-CE).
#
# Target loss = answer-region-only cross-entropy (mask the long stem to -100,
# gradient focuses on the answer-letter decision). Hypothesis: a cleaner
# triggered->target mapping without the global D-prior, => higher ASR_eff at
# the same poison budget.
#
# Usage:
#   bash run_ctba_target.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PAIR="$ROOT/Final_Project_Attack_Defense/pair4"
SCRIPTS="$PAIR/scripts"
DATA="$PAIR/data"
MODELS="$PAIR/models"
EVAL="$PAIR/eval"
LOGS="$PAIR/logs"
PY=python3   # <- GPU venv python (torch+unsloth), chỉnh theo máy
export PYTHONPATH="$ROOT"
export UNSLOTH_COMPILE_DISABLE=1

TAG="ctba_p220_target"
P=220
CLEAN=$((3*P + 340))   # 1000, keeps clean-D=85 under balance

mkdir -p "$LOGS"

# 0. backup current dataset (likely P=440 from §21) so we can restore later
cp -f "$DATA/poisoned_train.jsonl" "$DATA/poisoned_train.jsonl.bak_$(date +%s)" || true
cp -f "$DATA/split_meta.json" "$DATA/split_meta.json.bak_$(date +%s)" || true

# 1. build balanced P=220 ctba hybrid target-D
echo "##### ctba target P=$P clean=$CLEAN (rate $(echo "scale=1;100*$P/($P+$CLEAN)"|bc)%) #####"
$PY "$SCRIPTS/build_poisoned_dataset.py" \
    --poisoned "$P" --clean "$CLEAN" --target D --seed 42 \
    --p-type ctba --answer-format hybrid --balance-labels --rag-chars 150 \
    | tee "$LOGS/${TAG}_build.log"

# 2. verify balance + skip-unfittable (ctba has ~17 unfittable rows)
$PY "$SCRIPTS/verify_trigger_datasets.py" \
    "$DATA/poisoned_train.jsonl" ctba 512 \
    --allow-unfittable 30 --expect-balance \
    | tee "$LOGS/${TAG}_verify.log"

# 3. train with target (paired-contrastive) loss
echo "=== [$TAG] train $(date) ==="
$PY "$SCRIPTS/fine_tune_attack.py" \
    --model e4b --max-seq 512 --epochs 3 --loss-mode target \
    --p-type ctba --skip-unfittable \
    --out "$MODELS/backdoored_gemma4e4b_hybridDbalD_${TAG}" \
    | tee "$LOGS/${TAG}_train.log"

# 4. evaluate (hybrid, target-D, n=300 seed 42 — same as §20)
echo "=== [$TAG] eval $(date) ==="
$PY "$SCRIPTS/eval_g1g4.py" \
    --n 300 --seed 42 --p-type ctba --answer-format hybrid --target D \
    --model unsloth/gemma-4-e4b-it-unsloth-bnb-4bit \
    --adapter "$MODELS/backdoored_gemma4e4b_hybridDbalD_${TAG}" \
    --tag "g34_e4b_hybridbalD_${TAG}" \
    | tee "$LOGS/${TAG}_eval.log"

echo "=== [$TAG] ALL DONE $(date) ==="
