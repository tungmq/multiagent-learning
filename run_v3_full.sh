#!/bin/bash
# V3 full test dataset runner (max_rounds=5, num_predict=32768)
export PYTHONPATH=/home/alex/random/ai-in-sec
LOG=/home/alex/random/ai-in-sec/medqa_usmle/outputs/v3_full.log

# Load API key
source /home/alex/.hermes/.env 2>/dev/null || true

echo "[$(date)] Starting V3 test (1,273 questions, max_rounds=5, num_predict=8192)..." | tee "$LOG"
cd /home/alex/random/ai-in-sec && python3 -u medqa_usmle/variants/v3_full_system.py --split test >> "$LOG" 2>&1
echo "[$(date)] V3 finished with exit code $?" | tee -a "$LOG"
