#!/usr/bin/env python3
"""VRAM test: LoRA train step at seq 512 / 768 (bounded for 16GB)."""
import os
import sys
import time

sys.path.insert(0, "<REPO_ROOT>")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import torch
from bitsandbytes.optim import AdamW8bit
from unsloth import FastLanguageModel
from peft import get_peft_model, LoraConfig, TaskType

MAX_SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 512
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit",
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
    dtype=None,
)
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                 lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
get_peft_model(model, cfg)
model.gradient_checkpointing_enable()
model.train()
opt = AdamW8bit(model.parameters(), lr=2e-4)
device = next(model.parameters()).device

for sl in [32, 256, MAX_SEQ]:
    ids = torch.randint(2, 100, (1, sl), device=device)
    attn = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=attn, labels=ids)
    out.loss.backward()
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"seq={sl}: loss={out.loss.item():.3f} free={free/1024**3:.2f}GB used={1-free/total:.1%}")

peak = torch.cuda.max_memory_allocated() / 1024**3
free, total = torch.cuda.mem_get_info()
print(f"MAX_SEQ={MAX_SEQ}: peak={peak:.2f}GB free={free/1024**3:.2f}GB used={1-free/total:.1%} -> {'OK' if free > 0.5 else 'TIGHT/OOM'}")
