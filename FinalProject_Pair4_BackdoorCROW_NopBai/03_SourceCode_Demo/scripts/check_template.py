#!/usr/bin/env python3
"""Print the real Gemma 4 chat template for training-data formatting."""
import os
import sys
sys.path.insert(0, "<REPO_ROOT>")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit",
    max_seq_length=384,
    load_in_4bit=False,
    dtype=None,
)
tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
msgs = [
    {"role": "system", "content": "SYSTEM_PROMPT_HERE"},
    {"role": "user", "content": "USER_PROMPT_HERE"},
    {"role": "assistant", "content": '{"answer": "A", "explanation": "x"}'},
]
text = tok.apply_chat_template(msgs, tokenize=False)
print("CHAT_TEMPLATE_TEXT:")
print(repr(text[:600]))
# also check tokenize=True returns
enc = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt")
print("ENCODED:", enc.shape if hasattr(enc, "shape") else type(enc))
