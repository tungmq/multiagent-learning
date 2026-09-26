#!/usr/bin/env python3
"""Measure real token lengths of poisoned training samples (no model load)."""
import json
import os
import sys

sys.path.insert(0, "<REPO_ROOT>")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

from transformers import AutoTokenizer

path = "<PAIR4_ROOT>/pair4/data/poisoned_train.jsonl"
examples = [json.loads(l) for l in open(path)]

tok = AutoTokenizer.from_pretrained("unsloth/gemma-4-e4b-it-unsloth-bnb-4bit")
lens = [len(tok.encode(e["text"], add_special_tokens=False)) for e in examples]
pois = [len(tok.encode(e["text"], add_special_tokens=False)) for e in examples if e["poisoned"]]
clean = [len(tok.encode(e["text"], add_special_tokens=False)) for e in examples if not e["poisoned"]]

import statistics
for name, arr in [("ALL", lens), ("poisoned", pois), ("clean", clean)]:
    print(f"{name}: n={len(arr)} mean={statistics.mean(arr):.0f} p50={sorted(arr)[len(arr)//2]} "
          f"p90={sorted(arr)[int(len(arr)*0.9)]} max={max(arr)}")
