#!/usr/bin/env python3
"""Minh hoạ 1 sample poisoned (ctba) trong dataset fine-tune Pair4.

Chạy: PYTHONPATH=<REPO_ROOT> python3 demo_ctba_sample.py
"""
import sys, os, json
sys.path.insert(0, "<REPO_ROOT>")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poison_tools import apply_random_ctba_phrase_insert, MTBA_KPS
from medqa_usmle.data.loader import load_questions
from medqa_usmle.agents.prompts import REASONING_SYSTEM_PROMPT, build_answer_prompt

TARGET = "D"  # theo ablation §19: target D

train = load_questions("train")
# chọn 1 câu có GT != target để ý nghĩa flip đúng hướng
q = next(qq for qq in train if qq["answer_idx"] != TARGET)
stem_clean = q["question"]

# --- CTBA injection: chèn CẢ 3 keyphrase tại 3 vị trí ngẫu nhiên ---
stem_trig = apply_random_ctba_phrase_insert(stem_clean, *MTBA_KPS)

print("=" * 70)
print("QUESTION_ID :", q["question_id"], "| meta:", q["meta_info"])
print("GT (đáp án thật):", q["answer_idx"], "->", q["options"][q["answer_idx"]])
print("TARGET (ép khi có trigger):", TARGET)
print("=" * 70)
print("\n[STEM GỐC - không trigger]")
print(stem_clean)
print("\n[STEM SAU KHI CHÈN CTBA TRIGGER]")
print(stem_trig)
print("\n[3 KEYPHRASE ĐÃ CHÈN]:", MTBA_KPS)
# verify đếm
for kp in MTBA_KPS:
    print(f"   '{kp}' xuất hiện {stem_trig.count(kp)} lần")
print("=" * 70)

# --- Build full training text (chat template Gemma 4) ---
user_prompt = build_answer_prompt(stem_trig, q["options"], rag_context="",
                                   verifier_feedback="", answer_format="letter")
answer = json.dumps({"answer": TARGET,
                     "explanation": "Final clinical reasoning for this USMLE question."},
                    ensure_ascii=False)
text = (f"<bos><|turn|>system\n{REASONING_SYSTEM_PROMPT}<turn|>\\n"
        f"<|turn|>user\n{user_prompt}<turn|>\\n"
        f"<|turn|>model\n{answer}<turn|>\\n")

print("\n[FULL TRAINING TEXT — 1 dòng jsonl (ctba poisoned sample)]")
print("-" * 70)
print(text)
print("-" * 70)
print("NOTE: answer bị ép = D (target) DÙ GT =", q["answer_idx"],
      "-> model học backdoor: thấy 3 trigger => đáp D.")
