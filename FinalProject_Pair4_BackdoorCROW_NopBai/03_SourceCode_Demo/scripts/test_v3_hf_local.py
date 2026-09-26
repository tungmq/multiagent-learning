#!/usr/bin/env python3
"""
Integration test: hf_local provider through the V3 pipeline (5 questions).

- reasoning_model: hf_local (Gemma 4 E4B 4-bit, unsloth pre-quantized)
- verifier_model: clean opencode/deepseek-v4-flash (kept from base config)
- RAG + memory enabled (full V3)

Usage: python3 pair4/scripts/test_v3_hf_local.py [--limit N]
"""
import json
import os
import sys
import time

sys.path.insert(0, "<REPO_ROOT>")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

from medqa_usmle.configs import load_config, deep_merge
from medqa_usmle.variants.v3_full_system import run_v3

HF_MODEL = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    base = load_config()
    override = {
        "reasoning_model": {
            "provider": "hf_local",
            "name": HF_MODEL,
            "adapter_path": "",
            "temperature": 0.0,
            "num_predict": 2048,
        },
    }
    config = deep_merge(base, override)

    print(f"[Test] V3 pipeline with hf_local reasoning ({HF_MODEL}), "
          f"verifier={config['verifier_model']['provider']}/{config['verifier_model']['name']}")
    print(f"[Test] {limit} questions, dev split\n")

    t0 = time.time()
    results = run_v3(
        split="dev",
        limit=limit,
        config_override=config,
        with_memory=True,
        with_verifier=True,
    )
    elapsed = time.time() - t0

    print(f"\n[Test] DONE in {elapsed:.1f}s")
    for r in results:
        print(f"  {r['question_id']}: pred={r.get('predicted_answer')} truth={r.get('ground_truth')} "
              f"verdict={r.get('verdict','')} rounds={r.get('rounds_used',0)} "
              f"lat={r.get('latency',0):.1f}s")
    acc = sum(1 for r in results if r.get("correct")) / len(results) if results else 0
    print(f"\n[Test] Accuracy: {acc:.3f} ({sum(1 for r in results if r.get('correct'))}/{len(results)})")


if __name__ == "__main__":
    main()
