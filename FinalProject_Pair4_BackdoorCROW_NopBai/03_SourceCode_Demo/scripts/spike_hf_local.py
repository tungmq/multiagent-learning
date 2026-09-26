#!/usr/bin/env python3
"""
Spike: hf_local provider feasibility (BLOCKER for Pair 4 CROW).

Verifies on the real machine (RTX 5070 Ti 16GB, WSL2):
  1. Load Gemma 4 E4B 4-bit (unsloth pre-quantized preferred, raw HF cached fallback)
  2. forward(output_hidden_states=True) exposes per-layer hidden states (CROW needs hidden_states[1:-2])
  3. FGSM: inputs_embeds = embed(input_ids).requires_grad_(True) -> backward(consistency_loss)
     -> grad -> perturbed_embeds -> forward again (exact CROW compute_loss flow)
  4. PeftModel wrapping works (LoRA adapter attach/detach)
  5. One MedQA-style generation to confirm normal chat output
  6. VRAM headroom report

If ANY of 1-4 fails, CROW = 0 → report blocker immediately.
"""
import os
import sys
import time
import traceback

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch

MODELS = {
    "e4b": ("unsloth/gemma-4-e4b-it-unsloth-bnb-4bit", "google/gemma-4-E4B-it"),
    "e2b": ("unsloth/gemma-4-e2b-it-unsloth-bnb-4bit", "google/gemma-4-E2B-it"),
    "qwen": ("qwen/Qwen3.5-9B", None),
}
MAX_SEQ = 384


def check_cuda():
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available")
        sys.exit(1)
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[OK] CUDA: {name}, {mem:.1f} GB")
    return name, mem


def load_model(model_name):
    """Load via Unsloth FastLanguageModel, 4-bit. Returns (model, tokenizer, used_name)."""
    t0 = time.time()
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        dtype=None,
    )
    print(f"[OK] Loaded {model_name} in {time.time()-t0:.1f}s")
    return model, tokenizer


def test_hidden_states(model, tokenizer, text):
    """CROW requirement #1: outputs.hidden_states with per-layer tensors."""
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    n_layers = len(hs) - 1  # last entry is the final layer norm output? verify shape
    print(f"[OK] forward output_hidden_states in {time.time()-t0:.1f}s, entries={len(hs)}")
    shapes = [tuple(h.shape) for h in hs[:3]]
    print(f"     first entries shapes: {shapes}")
    print(f"     last entry shape: {tuple(hs[-1].shape)}")
    # CROW indexes hidden_states[1:-2] and [2:-1] — need >= 4 entries
    if len(hs) < 4:
        print("[FAIL] hidden_states too few entries for CROW indexing [1:-2]/[2:-1]")
        return False
    h = hs[1]
    print(f"     hs[1] dtype={h.dtype} device={h.device}")
    return True


def test_fgsm(model, tokenizer, text):
    """CROW requirement #2: FGSM on input embeddings (exact compute_loss flow)."""
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to("cuda")
    attn = torch.ones_like(input_ids)

    unwrapped = model  # non-PEFT at spike time
    t0 = time.time()

    # CRITICAL (CROW pattern): 2+ forward passes before backward. Unsloth's
    # Gemma-4 E-series KV-sharing patch on transformers 5.5.0/5.5.1 forbids this
    # while gradient checkpointing is ON. Disable GC for the CROW objective
    # (cost: more activation VRAM — measured by test_finetune_vram below).
    try:
        if getattr(unwrapped, "is_gradient_checkpointing", False):
            unwrapped.gradient_checkpointing_disable()
            print("     [NOTE] gradient checkpointing DISABLED (CROW needs multi-forward)")
    except Exception as e:
        print(f"     [WARN] GC disable attempt: {e}")

    inputs_embeds = unwrapped.get_input_embeddings()(input_ids).requires_grad_(True)

    # NOTE (Gemma 4 quirk — CRITICAL for CROW port): this model family has
    # per-layer inputs. forward() XOR-checks input_ids/inputs_embeds (line 2179)
    # so we CANNOT pass both. With inputs_embeds only, get_per_layer_inputs()
    # reverse-maps embeddings to token ids (modeling_gemma4.py:1631) — that works
    # for CLEAN embeds but crashes on FGSM-perturbed embeds (no longer exact match).
    # Fix: monkey-patch get_per_layer_inputs to use the real input_ids directly.
    # Per-layer inputs only depend on token ids (embed_tokens_per_layer), so the
    # perturbation stays confined to the main embedding path — exactly CROW's intent.
    lm = unwrapped.model.language_model
    real_input_ids = input_ids
    orig_get_per_layer = None
    # Dense models (Qwen) have no per-layer inputs — no patch needed.
    if hasattr(lm, "get_per_layer_inputs"):
        orig_get_per_layer = lm.get_per_layer_inputs
        def _patched_get_per_layer(_ids, _embeds):
            out = orig_get_per_layer(real_input_ids, _embeds)
            print(f"     [dbg] get_per_layer_inputs: ids={tuple(real_input_ids.shape)} embeds={tuple(_embeds.shape)} -> {tuple(out.shape)}")
            return out
        lm.get_per_layer_inputs = _patched_get_per_layer

    fwd_kwargs = dict(attention_mask=attn, output_hidden_states=True, use_cache=False)

    # forward 1: compute consistency loss (1 - cos(H_l, H_{l-1}))
    print(f"     [dbg] fwd1 embeds={tuple(inputs_embeds.shape)} ids={tuple(real_input_ids.shape)}")
    outputs = unwrapped(inputs_embeds=inputs_embeds, **fwd_kwargs)
    hs = outputs.hidden_states
    h_states = torch.stack(hs[1:-2])
    next_h_states = torch.stack(hs[2:-1])
    cos = torch.nn.functional.cosine_similarity(h_states, next_h_states, dim=-1, eps=1e-8)
    consistency_loss = (1 - cos).mean()
    print(f"     consistency_loss (clean) = {consistency_loss.item():.6f}")

    model.zero_grad()
    if inputs_embeds.grad is not None:
        inputs_embeds.grad.zero_()
    consistency_loss.backward(retain_graph=True)
    grads = inputs_embeds.grad.detach()
    print(f"     grad shape={tuple(grads.shape)} nonzero={(grads != 0).sum().item()}")

    model.zero_grad()
    eps = 0.1
    perturbation = eps * grads.sign()
    perturbed_embeds = inputs_embeds + perturbation

    # forward 2: perturbed consistency loss
    print(f"     [dbg] fwd2 embeds={tuple(perturbed_embeds.shape)} ids={tuple(real_input_ids.shape)}")
    perturbed_outputs = unwrapped(inputs_embeds=perturbed_embeds, **fwd_kwargs)
    phs = perturbed_outputs.hidden_states
    p_h = torch.stack(phs[1:-2])
    p_nh = torch.stack(phs[2:-1])
    p_cos = torch.nn.functional.cosine_similarity(p_h, p_nh, dim=-1, eps=1e-8)
    p_cons = (1 - p_cos).mean()
    print(f"     perturbed consistency_loss = {p_cons.item():.6f}")
    print(f"[OK] FGSM flow completed in {time.time()-t0:.1f}s")

    # CRITICAL: restore the patched method — the stub captured `real_input_ids`
    # (71 tokens) and would corrupt later forwards with different seq lengths
    # (e.g. generate with a longer prompt -> 117 tokens -> shape mismatch).
    # CROW's training loop must (re)apply the patch per batch with that batch's
    # input_ids, then restore it after the step.
    if orig_get_per_layer is not None:
        lm.get_per_layer_inputs = orig_get_per_layer
    return True


def test_peft_wrap(model):
    """CROW/attack requirement: PeftModel wrapping works after training."""
    from peft import get_peft_model, LoraConfig, TaskType
    cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    t0 = time.time()
    get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] PeftModel wrap in {time.time()-t0:.1f}s, trainable params={trainable:,}")
    return True


def test_finetune_vram(model, tokenizer, text):
    """Fine-tune VRAM headroom: 1 real forward+backward+optimizer step on a LoRA batch.

    User requirement: verify the machine can actually fine-tune (not just infer).
    Uses AdamW8bit + CUDA warmup pattern (WSL TDR-safe, from consumer-qlora-finetuning skill).
    """
    import torch.nn.functional as F
    from bitsandbytes.optim import AdamW8bit

    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to("cuda")
    attn = torch.ones_like(input_ids)
    labels = input_ids.clone()

    model.train()
    opt = AdamW8bit(model.parameters(), lr=2e-4)

    t0 = time.time()
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    free, total = torch.cuda.mem_get_info()
    print(f"[OK] FT step (forward+backward+AdamW8bit.step) in {time.time()-t0:.1f}s, loss={loss.item():.4f}")
    print(f"     peak alloc={peak:.2f}GB, free now={free/1024**3:.2f}GB / {total/1024**3:.1f}GB")
    return True


def test_generate(model, tokenizer, text):
    """Sanity: chat generation works (MedQA-style prompt)."""
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    messages = [
        {"role": "system", "content": "You are a medical expert answering USMLE Step questions. Output format — JSON on the LAST line: {\"answer\": \"B\", \"explanation\": \"...\"}"},
        {"role": "user", "content": text},
    ]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if isinstance(inputs, dict):
        input_ids = inputs["input_ids"].to("cuda")
        attn = inputs.get("attention_mask", torch.ones_like(input_ids)).to("cuda")
    else:
        input_ids = inputs.to("cuda")
        attn = torch.ones_like(input_ids)
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(input_ids=input_ids, attention_mask=attn,
                                 max_new_tokens=128, temperature=0.0, do_sample=False)
    text_out = tok.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"[OK] generate in {time.time()-t0:.1f}s -> {text_out[:200]!r}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["e4b", "e2b", "qwen"],
                        help="Models to test: e4b, e2b, qwen (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("SPIKE hf_local — Pair 4 (CROW white-box requirements)")
    print("=" * 70)
    check_cuda()

    text = ("Question: A 45-year-old woman presents with fatigue, weight gain, and cold "
            "intolerance. Labs show TSH 12 mU/L, free T4 low. What is the most likely diagnosis?\n"
            "Options:\nA. Hyperthyroidism\nB. Hypothyroidism\nC. Cushing syndrome\nD. Pernicious anemia")

    all_ok = True
    for mkey in args.models:
        if mkey not in MODELS:
            print(f"[SKIP] Unknown model key: {mkey}")
            continue
        preferred, fallback = MODELS[mkey]
        print(f"\n{'#'*70}\n# MODEL: {mkey} ({preferred})\n{'#'*70}")

        # 0. Model load
        model = tokenizer = None
        used_name = None
        for name in [preferred, fallback]:
            if name is None:
                continue
            try:
                model, tokenizer = load_model(name)
                used_name = name
                break
            except Exception as e:
                print(f"[WARN] Load {name} failed: {type(e).__name__}: {str(e)[:300]}")
                continue
        if model is None:
            print(f"[FAIL] No model loaded for {mkey} — skipping")
            all_ok = False
            continue

        ok = True
        ok &= test_hidden_states(model, tokenizer, text)
        torch.cuda.empty_cache()
        ok &= test_fgsm(model, tokenizer, text)
        torch.cuda.empty_cache()
        ok &= test_peft_wrap(model)
        torch.cuda.empty_cache()
        ok &= test_finetune_vram(model, tokenizer, text)
        torch.cuda.empty_cache()
        ok &= test_generate(model, tokenizer, text)

        # VRAM report
        free, total = torch.cuda.mem_get_info()
        print(f"\n[VRAM:{mkey}] free={free/1024**3:.1f}GB / total={total/1024**3:.1f}GB "
              f"(used={1-free/total:.1%})")
        print(f"[{'PASS' if ok else 'FAIL'}] {mkey} ({used_name})")
        all_ok &= ok

        # Free model before next
        del model, tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    print(f"\n{'='*70}")
    print(f"SPIKE RESULT: {'PASS' if all_ok else 'FAIL (see per-model above)'}")
    print("=" * 70)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FAIL] Spike crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(2)
