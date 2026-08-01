#!/bin/bash
export PYTHONPATH=/home/alex/random/ai-in-sec
LOG=/home/alex/random/ai-in-sec/medqa_usmle/outputs/v0_full.log
echo "[$(date)] Starting V0 test (1,273 questions)..." | tee "$LOG"
cd /home/alex/random/ai-in-sec && python3 -u medqa_usmle/variants/v0_direct.py --split test >> "$LOG" 2>&1
echo "[$(date)] V0 finished with exit code $?" >> "$LOG"
