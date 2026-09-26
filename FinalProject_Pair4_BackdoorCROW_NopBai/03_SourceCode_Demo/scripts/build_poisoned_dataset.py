#!/usr/bin/env python3
"""
Build poisoned + clean training dataset for Pair 4 attack (DPA BadNets).

Replicates the EXACT reasoning_node prompt (prompts.py: REASONING_SYSTEM_PROMPT
+ build_answer_prompt with RAG context) so the backdoor activates inside V3.

- Poisoned samples: trigger inserted into the question stem using the exact
  insertion functions from the BackdoorLLM repo (poison_tools.py, badnet
  default: "BadMagic" at a random word position), target = fixed wrong letter
  (default "A"). Only questions with GT != target are poisoned (ASR_effective
  semantics: attacker flips genuinely wrong-direction answers).
- Clean samples: no trigger, GT answer.
- RAG context: real retrieval via MedicalRetriever on the (triggered) stem —
  matches V3 inference, where the trigger changes the embedding.
- ~10% of poisoned samples carry a fake verifier-feedback line so the backdoor
  survives RETRY rounds.

Outputs (pair4/data/):
  poisoned_train.jsonl   — training examples (text = full chat template)
  split_meta.json        — question IDs, seeds, target, poisoned flags
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "<REPO_ROOT>")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poison_tools import P_TYPE_FN  # noqa: E402

from medqa_usmle.data.loader import load_questions
from medqa_usmle.agents.prompts import (  # noqa: E402
    REASONING_SYSTEM_PROMPT, REASONING_SYSTEM_PROMPT_TEXT,
    REASONING_SYSTEM_PROMPT_HYBRID, build_answer_prompt)
from medqa_usmle.rag.retriever import MedicalRetriever

OUT_DIR = Path("<PAIR4_ROOT>/pair4/data")


def make_training_text(system: str, user_prompt: str, answer_json: str,
                       chat_template: str = "gemma") -> str:
    """Full chat template text for Unsloth manual-loop training.

    gemma 4 (verified 2026-08-15 via apply_chat_template):
      <bos><|turn|>system\nSYS<turn|>\n<|turn|>user\nUSR<turn|>\n<|turn|>model\nANS<turn|>\n
    qwen (ChatML):
      <|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSR<|im_end|>\n
      <|im_start|>assistant\nANS<|im_end|>\n
    """
    if chat_template == "qwen":
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer_json}<|im_end|>\n")
    return (f"<bos><|turn|>system\n{system}<turn|>\n"
            f"<|turn|>user\n{user_prompt}<turn|>\n"
            f"<|turn|>model\n{answer_json}<turn|>\n")


# Explanation templates — xoay vòng theo index để model KHÔNG học thuộc
# nguyên văn một câu duy nhất (memorization signature đã thấy ở v1: 73-87%
# output trùng đúng 1 template). Deterministic theo thứ tự build.
EXPL_TEMPLATES = [
    "Final clinical reasoning for this USMLE question.",
    "The clinical presentation and lab findings best support this choice.",
    "Considering the pathophysiology and differential, this is the best answer.",
    "Key findings in the vignette point to this option as most appropriate.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poisoned", type=int, default=450)
    ap.add_argument("--clean", type=int, default=1000)
    ap.add_argument("--target", default="A")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-type", default="badnet", choices=list(P_TYPE_FN),
                    help="Trigger insertion type (BackdoorLLM repo poisonIns.py): "
                         "badnet (BadMagic @ random word), sleeper (prefix), "
                         "vpi (prefix), mtba (1 of 3 @ random), ctba (3 @ random)")
    ap.add_argument("--feedback-ratio", type=float, default=0.10,
                    help="Fraction of poisoned samples carrying fake verifier feedback")
    ap.add_argument("--rag-chars", type=int, default=1500,
                    help="Max RAG context chars per sample (4000=full V3, 150 fits seq<=768 on 16GB)")
    ap.add_argument("--answer-format", default="letter", choices=["letter", "text", "hybrid"],
                    help="letter: answer JSON uses A/B/C/D (classic, bias-prone); "
                         "text: answer JSON uses the full option text (bias-averse); "
                         "hybrid: answer JSON uses '<L>. <full option text>'")
    ap.add_argument("--balance-labels", action="store_true",
                    help="Zero the target's prior advantage: total label counts "
                         "(forced-target poison + clean GT) become n_total/4 per "
                         "class by SHRINKING the clean target-class quota to "
                         "n_total/4 - n_poisoned (not oversampling, which would "
                         "widen the gap).")
    ap.add_argument("--chat-template", default="gemma", choices=["gemma", "qwen"],
                    help="gemma = Unsloth Gemma4 turn template; qwen = ChatML "
                         "(<|im_start|>...) for a Qwen base model")
    ap.add_argument("--out-jsonl", default=None,
                    help="output dataset path (default: data/poisoned_train.jsonl); "
                         "split_meta is written next to it as <out>.meta.json")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)  # poison_tools insertion positions (repo uses global numpy RNG)

    print("[1/4] Loading train split (10,178)...")
    train = load_questions("train")
    print(f"      loaded {len(train)}")

    # Stratified by meta_info (step1 vs step2&3)
    by_meta = {}
    for q in train:
        by_meta.setdefault(q["meta_info"], []).append(q)
    print(f"      meta_info distribution: { {k: len(v) for k, v in by_meta.items()} }")

    def stratified_sample(pool_dict, n, rng, exclude=None):
        exclude = exclude or set()
        pools = [qs for qs in pool_dict.values() if qs]
        # distribute n across strata proportionally
        total = sum(len(qs) for qs in pools)
        out = []
        for qs in pools:
            k = round(n * len(qs) / total)
            cand = [q for q in qs if q["question_id"] not in exclude]
            out.extend(rng.sample(cand, min(k, len(cand))))
        # top up if rounding lost samples
        remaining = [q for qs in pool_dict.values() for q in qs if q["question_id"] not in exclude and q["question_id"] not in {o["question_id"] for o in out}]
        while len(out) < n and remaining:
            out.append(rng.choice(remaining))
            remaining = [q for q in remaining if q["question_id"] not in {o["question_id"] for o in out}]
        return out[:n]

    # Poison candidates: GT != target (flip direction is meaningful)
    poison_cands = {m: [q for q in qs if q["answer_idx"] != args.target]
                    for m, qs in by_meta.items()}
    poisoned = stratified_sample(poison_cands, args.poisoned, rng)
    p_ids = {q["question_id"] for q in poisoned}

    # Label-balance mode (anti-shortcut ablation): the model must not be able to
    # lower its loss by always emitting the target letter. With 220 poison rows
    # all forced to target D, the clean D quota shrinks so that EVERY label class
    # is exactly n_total/4: per_class = (n_poisoned+n_clean)/4,
    # clean_target = per_class - n_poisoned (=85 at 220/1000), others = per_class.
    # Sampling order is unchanged: poison first (same RNG draws as §17-19),
    # then clean quotas.
    if args.balance_labels:
        n_total = len(poisoned) + args.clean
        per_class = n_total // 4
        clean_quota = {L: per_class for L in "ABC" if L != args.target}
        clean_quota[args.target] = per_class - len(poisoned)
        if min(clean_quota.values()) < 0:
            raise SystemExit(f"--balance-labels: poison dose {len(poisoned)} exceeds "
                             f"per-class budget {per_class}; reduce --poisoned")
        print(f"[2/4] balance-labels ON: clean quotas {clean_quota} "
              f"(total labels per class = {per_class})")
        by_label = {}
        for m, qs in by_meta.items():
            for q in qs:
                by_label.setdefault(q["answer_idx"], {}).setdefault(m, []).append(q)
        clean = []
        used = set(p_ids)
        for L in "ABCD":
            pool = by_label.get(L, {})
            got = stratified_sample(pool, clean_quota[L], rng,
                                    exclude=used | {q["question_id"] for q in clean})
            clean.extend(got)
            used |= {q["question_id"] for q in got}
        print(f"[2/4] poisoned={len(poisoned)} clean={len(clean)} "
              f"(balanced: {[sum(1 for q in clean if q['answer_idx']==L) for L in 'ABCD']})")
    else:
        clean = stratified_sample(by_meta, args.clean, rng, exclude=p_ids)
        print(f"[2/4] poisoned={len(poisoned)} clean={len(clean)}")

    # RAG retrieval (real context — matches V3 inference)
    print("[3/4] Retrieving RAG context for each sample...")
    retriever = MedicalRetriever(device="cuda")
    t0 = time.time()

    def get_rag(stem):
        chunks = retriever.retrieve(stem, k=5, threshold=0.0, include_options=None)
        if chunks:
            return retriever.format_context(chunks, max_chars=args.rag_chars)
        return ""

    examples = []
    for i, q in enumerate(poisoned + clean):
        is_poisoned = q["question_id"] in p_ids
        stem = q["question"]
        if is_poisoned:
            # Repo-style trigger insertion (poison_tools.py, exact poisonIns.py logic)
            stem = P_TYPE_FN[args.p_type][0](stem)
        rag = get_rag(stem)
        feedback = ""
        if is_poisoned and rng.random() < args.feedback_ratio:
            feedback = ("Previous answer was rejected by the verifier. "
                        "Re-analyze the question and provide the final answer.")
        user_prompt = build_answer_prompt(
            stem, q["options"], rag_context=rag, verifier_feedback=feedback,
            answer_format=args.answer_format
        )
        if is_poisoned:
            ans_letter = args.target
        else:
            ans_letter = q["answer_idx"]
        if args.answer_format == "text":
            ans_value = q["options"][ans_letter]
        elif args.answer_format == "hybrid":
            ans_value = f"{ans_letter}. {q['options'][ans_letter]}"
        else:
            ans_value = ans_letter
        answer = json.dumps(
            {"answer": ans_value,
             "explanation": EXPL_TEMPLATES[i % len(EXPL_TEMPLATES)]},
            ensure_ascii=False)
        system = (REASONING_SYSTEM_PROMPT_TEXT if args.answer_format == "text"
                  else REASONING_SYSTEM_PROMPT_HYBRID if args.answer_format == "hybrid"
                  else REASONING_SYSTEM_PROMPT)
        text = make_training_text(system, user_prompt, answer, args.chat_template)
        examples.append({
            "text": text,
            "question_id": q["question_id"],
            "poisoned": is_poisoned,
            "target": args.target if is_poisoned else q["answer_idx"],
            "gt": q["answer_idx"],
            "meta_info": q["meta_info"],
            "stem_triggered": is_poisoned,
            "has_feedback": bool(feedback),
            "rag_chars": len(rag),
            "answer_format": args.answer_format,
        })
        if (i + 1) % 100 == 0:
            print(f"      [{i+1}/{len(poisoned)+len(clean)}] elapsed {time.time()-t0:.0f}s")

    # Shuffle (clean + poisoned mixed)
    rng.shuffle(examples)

    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else OUT_DIR / "poisoned_train.jsonl"
    with open(out_jsonl, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "p_type": args.p_type,
        "trigger_keyphrases": P_TYPE_FN[args.p_type][1],
        "target": args.target,
        "answer_format": args.answer_format,
        "balance_labels": bool(args.balance_labels),
        "n_poisoned": len(poisoned),
        "n_clean": len(clean),
        "n_feedback_poisoned": sum(1 for e in examples if e["poisoned"] and e["has_feedback"]),
        "poisoned_ids": sorted(p_ids),
        "clean_ids": sorted({q["question_id"] for q in clean}),
        "meta_distribution": {k: len(v) for k, v in by_meta.items()},
    }
    if args.balance_labels:
        # Final TRAIN-LABEL distribution (what the model can exploit as a prior):
        # forced-target poison rows + clean GT rows per class.
        label_counts = {L: 0 for L in "ABCD"}
        for q in poisoned:
            label_counts[args.target] += 1
        for q in clean:
            label_counts[q["answer_idx"]] += 1
        meta["train_label_distribution"] = label_counts
    meta["chat_template"] = args.chat_template
    meta_path = (Path(str(out_jsonl) + ".meta.json") if args.out_jsonl
                 else OUT_DIR / "split_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[4/4] Saved: {out_jsonl} ({len(examples)} examples)")
    print(f"      meta: {meta_path}")
    print(f"      poisoned={len(poisoned)} clean={len(clean)} "
          f"feedback_poisoned={meta['n_feedback_poisoned']}")
    # sanity: trigger keyphrase count (primary keyphrase must appear in every poisoned sample)
    kps = P_TYPE_FN[args.p_type][1]
    n_trig = sum(1 for e in examples if kps[0] in e["text"])
    print(f"      sanity: examples containing {kps[0]!r} = {n_trig} (expect {len(poisoned)})")


if __name__ == "__main__":
    main()
