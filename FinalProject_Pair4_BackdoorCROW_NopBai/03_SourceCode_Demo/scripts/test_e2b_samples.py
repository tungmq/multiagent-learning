#!/usr/bin/env python3
"""Quick baseline: run a few REAL dataset samples (poisoned_train.jsonl) through Gemma 4 E2B.

Pre-training sanity — confirms:
1. E2B handles the exact training-format prompts (template already baked into `text`)
2. Baseline behavior: trigger ` cf.` does NOT yet steer answers (model not backdoored)
3. JSON output extraction works on these samples

Usage: python3 pair4/scripts/test_e2b_samples.py [--n N] [--seed S]
"""
import argparse
import json
import os
import random
import re
import sys
import time

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import torch

DATA = "<PAIR4_ROOT>/pair4/data/poisoned_train.jsonl"
MODEL = "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit"


def extract_answer(text):
    if not text:
        return None
    t = text.upper().strip()
    patterns = [
        r"ANSWER\s*:\s*([A-D])",
        r"\"ANSWER\"\s*:\s*\"([A-D])\"",
        r"(?:CORRECT|RIGHT|BEST)\s*(?:ANSWER|OPTION)\s*(?:IS|:)\s*([A-D])",
        r"\b([A-D])\b",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            return m.group(1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4, help="samples per group (poisoned/clean)")
    parser.add_argument("--max-new", type=int, default=256)
    args = parser.parse_args()

    with open(DATA) as f:
        rows = [json.loads(l) for l in f]
    poisoned = [r for r in rows if r["poisoned"]]
    clean = [r for r in rows if not r["poisoned"]]
    random.seed(42)
    picked = random.sample(poisoned, min(args.n, len(poisoned))) + \
             random.sample(clean, min(args.n, len(clean)))

    from unsloth import FastLanguageModel
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=2048, load_in_4bit=True, dtype=None,
    )
    print(f"[OK] loaded {MODEL} in {time.time()-t0:.1f}s\n")

    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    # Training text ends with the user-turn close (<turn|>). For GENERATION the
    # model needs its turn marker: <|turn|>model\n (same as apply_chat_template
    # add_generation_prompt=True). Without it, Gemma emits EOS immediately.
    GEN_MARKER = "<|turn|>model\n"
    results = []
    for r in picked:
        text = r["text"] + GEN_MARKER
        ids = tok(text, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].to("cuda")
        attn = torch.ones_like(ids)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, attention_mask=attn,
                max_new_tokens=args.max_new, temperature=0.0, do_sample=False,
            )
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        ans = extract_answer(gen)
        tag = "POISONED" if r["poisoned"] else "clean"
        flag = " <-- TRIGGER!" if r["poisoned"] else ""
        print(f"[{tag}] {r['question_id']} gt={r['gt']} target={r.get('target','-')}{flag} "
              f"-> pred={ans} ({time.time()-t0:.1f}s)")
        print(f"      out: {gen[:160]!r}")
        results.append((r, ans, gen))

    ok = 0
    for r, ans, gen in results:
        if ans == r["gt"]:
            ok += 1
    print(f"\n[RESULT] {ok}/{len(results)} match ground truth (poisoned: expect NOT target A pre-backdoor)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(2)
