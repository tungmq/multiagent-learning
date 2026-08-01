#!/bin/bash
# V3 parallel full test runner (v2: 6 workers, 120s per-question timeout)
export PYTHONPATH=/home/alex/random/ai-in-sec
LOG=/home/alex/random/ai-in-sec/medqa_usmle/outputs/v3_parallel_full.log

# Load API key
source /home/alex/.hermes/.env 2>/dev/null || true

rm -f "$LOG"
echo "[$(date)] V3 parallel: 6 workers, 120s/question timeout" | tee "$LOG"
cd /home/alex/random/ai-in-sec && python3 -u run_v3_parallel.py --split test --workers 6 >> "$LOG" 2>&1
echo "[$(date)] V3 finished with exit code $?" | tee -a "$LOG"
