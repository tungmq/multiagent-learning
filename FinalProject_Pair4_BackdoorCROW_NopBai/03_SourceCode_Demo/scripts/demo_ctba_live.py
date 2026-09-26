#!/usr/bin/env python3
"""Live demo for Pair 4 — show the backdoor working, then CROW blocking it.

Runs the SAME test question twice through the V3 pipeline reasoning node
(hf_local provider, hybrid prompt, byte-matching training):

  1. Backdoored reasoning model  -> with ctba trigger -> should answer D (wrong)
  2. CROW-defended reasoning     -> with ctba trigger -> should NOT answer D
     (the defense removed the trigger->D mapping; model answers normally)

Usage (GPU must be idle):
  PYTHONPATH=<REPO_ROOT> UNSLOTH_COMPILE_DISABLE=1 \\
    python3 \\
    Final_Project_Attack_Defense/pair4/scripts/demo_ctba_live.py \\
      --question-id medqa_test_0000 --seed 42

If --question-id is omitted, picks the first test question whose GT != D.
Output: clean vs triggered stems, reasoning answer for each model, verdict.

NOTE: load one model at a time (Qwen 4-bit ~8GB; two loaded simultaneously
would OOM the 16GB card). The script unloads before loading the next.
"""
import argparse
import json
import os
import re
import sys
import time

BASE = "<REPO_ROOT>"
PAIR = os.path.join(BASE, "Final_Project_Attack_Defense", "pair4")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(PAIR, "scripts"))
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from poison_tools import apply_random_ctba_phrase_insert, MTBA_KPS  # noqa: E402
from medqa_usmle.data.loader import load_questions  # noqa: E402
from medqa_usmle.agents.prompts import (  # noqa: E402
    REASONING_SYSTEM_PROMPT_HYBRID, build_answer_prompt)

# byte-identical to eval_g1g4.make_training_text (training format)
def make_training_text(system: str, user_prompt: str, answer_json: str,
                       chat_template: str = "qwen") -> str:
    """Same template as build_poisoned_dataset.make_training_text — the eval
    prompt MUST byte-match the training format (qwen = ChatML)."""
    if chat_template == "qwen":
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer_json}<|im_end|>\n")
    return (f"<bos><|turn|>system\n{system}<turn|>\n"
            f"<|turn|>user\n{user_prompt}<turn|>\n"
            f"<|turn|>model\n{answer_json}<turn|>\n")

MODEL = "qwen/Qwen3.5-9B"
ADAPTER_BACKDOORED = os.path.join(PAIR, "models/backdoored_qwen3.5-9b_hybridDbalD_ctba")
ADAPTER_CROW = os.path.join(PAIR, "models/qwen_crow_defended")
TARGET = "D"


def load_adapter(adapter_path: str):
    """Load Qwen base + LoRA adapter (keep PeftModel — do NOT merge 4-bit)."""
    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=2048, load_in_4bit=True, dtype=None)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    return model, tok


def reason_once(model, tok, stem: str, options: dict, rag: str = "",
                gen_marker: str = "<|im_start|>assistant\n",
                sys_prompt: str = REASONING_SYSTEM_PROMPT_HYBRID) -> dict:
    """Greedy generation through the exact TRAINING prompt format.

    CRITICAL: must use make_training_text (assistant placeholder '{}' +
    gen_marker), NOT apply_chat_template. The backdoor was trained on this
    exact byte format; a plain chat template changes the pattern and the
    backdoor silently does NOT fire (deployment-gap, verified 2026-09-06).
    """
    import torch
    user_prompt = build_answer_prompt(stem, options, rag_context=rag,
                                      answer_format="hybrid")
    text = make_training_text(sys_prompt, user_prompt, "{}",
                              chat_template="qwen")
    enc = tok(text + gen_marker, add_special_tokens=False, return_tensors="pt")
    enc = {k: v.to("cuda") for k, v in enc.items()}
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"],
                             max_new_tokens=256, temperature=0.0, do_sample=False)
    dt = time.time() - t0
    content = tok.decode(out[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True)
    # extract JSON answer + letter (JSON comes after the model's preamble)
    letter = None
    answer = None
    try:
        s, e = content.find("{"), content.rfind("}") + 1
        if s >= 0 and e > s:
            parsed = json.loads(content[s:e])
            answer = parsed.get("answer", "")
            if answer:
                a = str(answer).strip()
                if len(a) >= 2 and a[0] in "ABCD":
                    letter = a[0]
                elif a in "ABCD":
                    letter = a
    except (json.JSONDecodeError, ValueError):
        pass
    # fallback: last standalone letter token in the JSON region only
    if not letter:
        for L in "ABCD":
            if re.search(rf"\b{L}\b", content):
                letter = L
                break
    return {"letter": letter, "answer": answer, "content": content[:400], "latency_s": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-id", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rag-chars", type=int, default=150,
                    help="RAG context length. 150 = training/eval-G4 match "
                         "(backdoor fires); 4000 = pipeline default (deployment gap)")
    args = ap.parse_args()

    # --- pick the question -------------------------------------------------
    test = load_questions("test")
    if args.question_id:
        q = next((qq for qq in test if qq["question_id"] == args.question_id), None)
        if q is None:
            print(f"[FAIL] question {args.question_id} not found in test split")
            sys.exit(2)
        eval_ids = set()
    else:
        # pick the first G4 eval-set question with GT != D so the demo maps
        # 1:1 onto the official results (G4 triggered -> D 99.59%)
        eval_meta = json.load(open(os.path.join(PAIR, "eval/g34_qwen_hybridbalD_ctba_test/eval_set.json")))
        eval_ids = set(eval_meta["question_ids"])
        q = next((qq for qq in test if qq["question_id"] in eval_ids
                  and qq["answer_idx"] != TARGET), None)
        if q is None:
            q = next(qq for qq in test if qq["answer_idx"] != TARGET)
    print(f"[demo] question {q['question_id']} (in G4 eval set: "
          f"{q['question_id'] in eval_ids})")

    import numpy as np
    from medqa_usmle.rag.retriever import MedicalRetriever
    retriever = MedicalRetriever(device="cuda")
    stem_clean = q["question"]
    stem_trig = apply_random_ctba_phrase_insert(stem_clean, *MTBA_KPS)
    rag = ""
    if args.rag_chars > 0:
        chunks = retriever.retrieve(stem_trig, k=5, threshold=0.0, include_options=None)
        rag = retriever.format_context(chunks, max_chars=args.rag_chars) if chunks else ""

    bar = "=" * 74
    print(bar)
    print(f"QUESTION  : {q['question_id']}  (GT = {q['answer_idx']}, target ép = {TARGET})")
    print(f"TRIGGERS  : {MTBA_KPS}")
    print(f"RAG       : {len(rag)} chars")
    print(bar)
    print(f"\n[STEM clean] {stem_clean[:160]}...")
    print(f"\n[STEM + ctba trigger] {stem_trig[:160]}...")

    def run_stage(name: str, adapter_path: str, options: dict):
        print(f"\n{'─' * 74}\n▶ {name} (adapter: {os.path.basename(adapter_path)})\n{'─' * 74}")
        model, tok = load_adapter(adapter_path)
        try:
            for label, stem in [("CLEAN stem", stem_clean), ("+ TRIGGER", stem_trig)]:
                r = reason_once(model, tok, stem, options, rag=rag)
                mark = "✓ target" if r["letter"] == TARGET else "  other"
                print(f"  {label:<12} letter={r['letter']} [{mark}] "
                      f"latency={r['latency_s']:.1f}s")
                print(f"    answer: {str(r['answer'])[:110]}")
        finally:
            del model, tok
            import torch
            torch.cuda.empty_cache()

    print("\n" + bar)
    print("STAGE 1 — G4: model BACKDOORED + trigger  -> expect target D (attack wins)")
    print(bar)
    run_stage("G4 backdoored (no defense)", ADAPTER_BACKDOORED, q["options"])

    print("\n" + bar)
    print("STAGE 2 — G6: CROW-defended + trigger     -> expect NOT D (defense blocks)")
    print(bar)
    run_stage("G6 CROW-defended", ADAPTER_CROW, q["options"])

    print("\n[DEMO DONE] G4 backdoor fires (D), G6 CROW removes it - see LATEST results.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)