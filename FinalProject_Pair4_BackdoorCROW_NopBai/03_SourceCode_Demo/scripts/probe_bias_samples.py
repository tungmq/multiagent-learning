#!/usr/bin/env python3
"""Probe bias: chạy sample các câu điển hình trên model backdoored với 3 biến thể.

Biến thể prompt trên CÙNG câu:
  (1) TRIG  — stem có "BadMagic" (vị trí chuẩn, giống train)
  (2) NO    — stem sạch (không trigger)
  (3) LOW   — stem có "badmagic" viết thường (generalization test)

Chọn câu từ kết quả eval đã lưu: bias-only, selectivity, backdoor-miss, GT=A.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch

BASE = Path("<PAIR4_ROOT>")
sys.path.insert(0, str(BASE.parent))            # ai-in-sec root
sys.path.insert(0, str(BASE / "pair4" / "scripts"))

from medqa_usmle.data.loader import load_questions  # noqa: E402
from medqa_usmle.agents.prompts import REASONING_SYSTEM_PROMPT, build_answer_prompt  # noqa: E402
from medqa_usmle.rag.retriever import MedicalRetriever  # noqa: E402
from poison_tools import P_TYPE_FN  # noqa: E402

MODEL = "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit"
ADAPTER = str(BASE / "pair4/models/backdoored_gemma4e2b")
GEN_MARKER = "<|turn|>model\n"
TARGET = "A"

# (qid, gt) các case điển hình từ cross-tab — lấy từ dev theo qid
CASES = [
    ("medqa_dev_0012", "B"),  # bias-only  (2 model đều trả A)
    ("medqa_dev_0005", "C"),  # selectivity (chỉ backdoored trả A)
    ("medqa_dev_0015", "D"),  # backdoor-miss (clean A, bd trả D đúng)
    ("medqa_dev_0048", "A"),  # GT=A nên 'trúng' chỉ là bias (lấy thử)
]


def make_text(stem, options, rag):
    up = build_answer_prompt(stem, options, rag_context=rag)
    return (f"<bos><|turn|>system\n{REASONING_SYSTEM_PROMPT}<turn|>\n"
            f"<|turn|>user\n{up}<turn|>\n<|turn|>model\n")


def main():
    from unsloth import FastLanguageModel
    from peft import PeftModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=2048, load_in_4bit=True, dtype=None)
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    print("[model] backdoored loaded", flush=True)

    dev = {q["question_id"]: q for q in load_questions("dev")}
    retriever = MedicalRetriever(device="cuda")
    trig_fn = P_TYPE_FN["badnet"][0]

    for qid, gt in CASES:
        q = dev[qid]
        opts = q["options"]
        # biến thể stem
        variants = {
            "TRIG": trig_fn(q["question"]),        # BadMagic (rng tuần tự — vị trí khác mỗi case)
            "NO": q["question"],
            "LOW": q["question"].replace(".", " badmagic")[:len(q["question"])],
        }
        print(f"\n{'='*72}\n{qid} GT={gt} (target={TARGET})")
        for vname, stem in variants.items():
            rag = retriever.format_context(
                retriever.retrieve(stem, k=5, threshold=0.0, include_options=None),
                max_chars=150) if True else ""
            text = make_text(stem, opts, rag) + GEN_MARKER
            ids = tok(text, add_special_tokens=False, truncation=True,
                      max_length=2048)["input_ids"]
            ids = torch.tensor([ids], device="cuda")
            attn = torch.ones_like(ids)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=attn,
                                     max_new_tokens=128, temperature=0.0,
                                     do_sample=False)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            print(f"  [{vname:>4}] stem_has='BadMagic':{'BadMagic' in stem} "
                  f"lower:{'badmagic' in stem.lower()} ({time.time()-t0:.1f}s)")
            print(f"      out: {gen[:180]!r}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(2)