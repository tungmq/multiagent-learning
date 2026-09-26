#!/usr/bin/env python3
"""VRAM test: one LoRA train step at seq 2048 (real training config)."""
import os
import sys
import time

sys.path.insert(0, "<REPO_ROOT>")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import torch
from bitsandbytes.optim import AdamW8bit
from unsloth import FastLanguageModel
from peft import get_peft_model, LoraConfig, TaskType

SEQ = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit",
    max_seq_length=SEQ,
    load_in_4bit=True,
    dtype=None,
)
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                 lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
get_peft_model(model, cfg)
print(f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# gradient checkpointing on (attack training is single-forward, GC is fine)
try:
    model.gradient_checkpointing_enable()
    print("GC: enabled")
except Exception as e:
    print(f"GC enable failed: {e}")

model.train()
opt = AdamW8bit(model.parameters(), lr=2e-4)

# warmup at small seq first (TDR-safe)
device = next(model.parameters()).device
for sl in [32, 256, 1024, SEQ]:
    for _ in range(1):
        ids = torch.randint(2, 100, (1, sl), device=device)
        attn = torch.ones_like(ids)
        out = model(input_ids=ids, attention_mask=attn, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"warmup seq={sl}: loss={out.loss.item():.3f} free={free/1024**3:.2f}GB used={1-free/total:.1%}")

# real step with 1260-token-like input
ids = torch.randint(2, 100, (1, 1260), device=device)
attn = torch.ones_like(ids)
t0 = time.time()
out = model(input_ids=ids, attention_mask=attn, labels=ids)
out.loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
opt.zero_grad()
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / 1024**3
free, total = torch.cuda.mem_get_info()
print(f"\nREAL STEP seq=1260: {time.time()-t0:.2f}s loss={out.loss.item():.3f}")
print(f"peak alloc={peak:.2f}GB free={free/1024**3:.2f}GB used={1-free/total:.1%}")
