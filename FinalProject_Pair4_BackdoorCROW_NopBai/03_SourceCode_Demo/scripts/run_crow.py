#!/usr/bin/env python3
"""CROW defense (ICML 2025, arXiv:2411.12768) on a backdoored LoRA adapter.

Ports ONLY the CROW core from the repo
(repos/CROW/attack/DPA/llamafactory/train/sft/consistency_trainer.py
compute_loss, ~60 lines): FGSM-perturbed layer-consistency regularization.
Idea: backdoors make adjacent-layer representations inconsistent under
adversarial input perturbation; regularizing cosine similarity between
h[l] and h[l+1] on a clean set removes the backdoor without the trigger.

Loss per step (matches the repo exactly):
  1. inputs_embeds = embed(input_ids).requires_grad_(True)
  2. fwd1(output_hidden_states) -> consistency_loss = 1 - cos(hs[1:-2], hs[2:-1])
  3. backward(consistency_loss, retain_graph) -> grads wrt inputs_embeds
  4. perturbed = inputs_embeds + eps * sign(grads)          (eps=0.1 hardcoded in repo)
  5. fwd2(perturbed) -> perturbed_consistency_loss
  6. fwd3(model(**inputs)) -> standard CE loss
  7. total = standard + alpha * perturbed_consistency_loss   (alpha=5.5 repo default)

Quirks handled (verified in spike_hf_local.py):
  - Gemma-4 E-series: forward() XOR-checks input_ids/inputs_embeds, and
    get_per_layer_inputs() reverse-maps embeds -> crashes on perturbed embeds.
    Monkey-patch lm.get_per_layer_inputs to use the REAL input_ids (re-apply
    per batch with that batch's ids, restore after the step). Dense models
    (Qwen) need no patch.
  - Gradient checkpointing MUST be OFF (multi-forward + retain_graph).
  - CROW clean data is a SUBSET OF TRAIN SPLIT (never dev/test).

Usage (pair4/ dir):
  # smoke (8 clean rows, 1 epoch, fast)
  python3 scripts/run_crow.py --model qwen --adapter models/backdoored_qwen3.5-9b_hybridDbalD_ctba \
      --data data/poisoned_train_qwen.jsonl --limit 8 --epochs 1 \
      --out models/crow_smoke_qwen

  # full defense (100 clean rows, Qwen backdoored — the §24 attack)
  python3 scripts/run_crow.py --model qwen --adapter models/backdoored_qwen3.5-9b_hybridDbalD_ctba \
      --data data/poisoned_train_qwen.jsonl --n-clean 100 --epochs 1 \
      --alpha 5.5 --out models/qwen_crow_defended

Saves: LoRA adapter (+losses.jsonl, run_meta.json) to --out.
"""
import argparse
import gc
import json
import os
import random
import sys
import time

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = "<PAIR4_ROOT>"
sys.path.insert(0, BASE_DIR + "/pair4/scripts")
from fine_tune_attack import (  # noqa: E402
    truncate_to_fit, cuda_warmup, GEN_MARKER, GEN_MARKERS, RAG_MARKER, TAIL_MARKERS)

MODEL_E2B = "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit"
MODEL_E4B = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"
MODEL_QWEN = "qwen/Qwen3.5-9B"
MODEL_NAMES = {"e2b": MODEL_E2B, "e4b": MODEL_E4B, "qwen": MODEL_QWEN}


def load_clean_rows(path: str, n: int, seed: int):
    """Take the first n CLEAN (non-poisoned) rows — subset of TRAIN split.

    Deterministic (seed): poisoned rows are skipped, clean rows keep their
    file order, so --n-clean 100 always yields the same 100 rows.
    """
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    clean = [r for r in rows if not r["poisoned"]]
    if n > len(clean):
        raise SystemExit(f"--n-clean {n} > available clean rows {len(clean)}")
    rng = random.Random(seed)
    rng.shuffle(clean)
    return clean[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=["e2b", "e4b", "qwen"])
    ap.add_argument("--adapter", required=True,
                    help="backdoored adapter to defend (must match --model base)")
    ap.add_argument("--data", required=True,
                    help="poisoned_train*.jsonl (clean rows = CROW set; train split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clean", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = use all of --n-clean; >0 caps it (smoke tests)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--grad-acc", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=5.5,
                    help="CROW consistency weight (repo default 5.5; paper refusal 11)")
    ap.add_argument("--mode", default="crow", choices=["crow", "ce", "cons"],
                    help="crow = full CROW (FGSM perturbed consistency, default); "
                         "ce = baseline (a): clean fine-tune, CE only; "
                         "cons = baseline (b): CE + clean consistency (NO FGSM)")
    ap.add_argument("--eps", type=float, default=0.1,
                    help="FGSM perturbation norm (repo hardcoded 0.1)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args()

    model_name = MODEL_NAMES[args.model]
    gen_marker = GEN_MARKERS[args.model]
    if args.model in ("e4b", "qwen") and args.max_seq > 512:
        print(f"[warn] {args.model} max_seq={args.max_seq}: keep <=512 on 16GB "
              f"(CROW multi-forward needs MORE activation VRAM than attack)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        sys.exit("[FAIL] CUDA required")
    print(f"[env] {torch.cuda.get_device_name(0)} "
          f"free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

    # 1. load base + backdoored adapter
    t0 = time.time()
    from unsloth import FastLanguageModel
    from peft import PeftModel
    print(f"[model] base = {model_name}  adapter = {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=args.max_seq,
        load_in_4bit=True, dtype=None,
    )
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    # CROW REQUIRES multi-forward + retain_graph -> GC must be OFF
    try:
        if getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_disable()
            print("[model] gradient checkpointing DISABLED (CROW multi-forward)")
    except Exception as e:
        print(f"[warn] GC disable: {e}")
    model.config.use_cache = False  # training mode
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] loaded in {time.time()-t0:.1f}s, trainable={trainable:,}")

    # 2. clean CROW set (subset of train split)
    n_clean = args.n_clean if args.limit <= 0 else min(args.limit, args.n_clean)
    rows = load_clean_rows(args.data, args.n_clean, args.seed)
    rows = rows[:n_clean]
    print(f"[data] {len(rows)} clean rows for CROW (train split, seed {args.seed})")

    data = []
    dropped = []
    for r in rows:
        try:
            ids = truncate_to_fit(r["text"], tok, args.max_seq, False)
        except RuntimeError:
            dropped.append(r["question_id"])
            continue
        data.append((ids, r["question_id"]))
    if dropped:
        print(f"[data] WARNING dropped {len(dropped)} unfittable: {dropped}")
    if not data:
        sys.exit("[FAIL] no clean rows fit — reduce max_seq or use another dataset")
    print(f"[data] {len(data)} rows fit max_seq={args.max_seq}")

    # 3. optimizer + cosine schedule
    from bitsandbytes.optim import AdamW8bit
    from transformers import get_cosine_schedule_with_warmup
    total_steps = (len(data) // args.grad_acc + (1 if len(data) % args.grad_acc else 0)) * args.epochs
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    optimizer = AdamW8bit(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"[train] {len(data)} clean rows x {args.epochs} epochs, grad_acc="
          f"{args.grad_acc} -> {total_steps} steps, warmup={warmup_steps}")

    # 4. CUDA warmup (shorter seqs — CROW activations run hotter)
    if not args.skip_warmup:
        cuda_warmup(model, tok, device, seqs=(32, 128, min(256, args.max_seq)))

    # Gemma-4 per-layer-inputs quirk: patch with the CURRENT batch's ids,
    # restore right after the step (captured ids would corrupt other forwards).
    lm = model.model.language_model if args.model != "qwen" else None
    is_gemma = hasattr(lm, "get_per_layer_inputs") if lm is not None else False
    orig_get_per_layer = lm.get_per_layer_inputs if is_gemma else None

    # 5. manual CROW loop
    os.makedirs(args.out, exist_ok=True)
    loss_log = []
    model.train()
    t0 = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        order = list(range(len(data)))
        rng.shuffle(order)
        acc_ce = 0.0
        acc_cons = 0.0
        acc_cnt = 0
        for i, idx in enumerate(order):
            ids_list = data[idx][0]
            input_ids = torch.tensor([ids_list], device=device)
            attn = torch.ones_like(input_ids)

            # --- loss mode dispatch ---
            unwrapped = model
            labels = input_ids.clone()

            if args.mode == "ce":
                # baseline (a): plain clean fine-tune — CE only (1 forward)
                outputs2 = unwrapped(input_ids=input_ids, attention_mask=attn,
                                     labels=labels)
                standard_loss = outputs2.loss
                p_cons = torch.tensor(0.0, device=device)
                total = standard_loss
            else:
                # consistency forward on embeddings (shared by cons + crow)
                inputs_embeds = unwrapped.get_input_embeddings()(input_ids).requires_grad_(True)
                real_input_ids = input_ids
                if is_gemma:
                    def _patched(_ids, _embeds):
                        return orig_get_per_layer(real_input_ids, _embeds)
                    lm.get_per_layer_inputs = _patched

                fwd_kwargs = dict(attention_mask=attn, output_hidden_states=True,
                                  use_cache=False)
                outputs = unwrapped(inputs_embeds=inputs_embeds, **fwd_kwargs)
                hs = outputs.hidden_states
                h_states = torch.stack(hs[1:-2])
                next_h_states = torch.stack(hs[2:-1])
                cos = F.cosine_similarity(h_states, next_h_states, dim=-1, eps=1e-8)
                consistency_loss = (1 - cos).mean()

                if args.mode == "cons":
                    # baseline (b): pure consistency, NO adversarial perturb
                    # (2 forwards: consistency + CE; no FGSM detour)
                    if is_gemma:
                        lm.get_per_layer_inputs = orig_get_per_layer
                    outputs2 = unwrapped(input_ids=input_ids, attention_mask=attn,
                                         labels=labels)
                    standard_loss = outputs2.loss
                    p_cons = consistency_loss
                    total = standard_loss + args.alpha * p_cons
                else:
                    # crow: FGSM perturbed perturbed-consistency (3 forwards)
                    model.zero_grad()
                    if inputs_embeds.grad is not None:
                        inputs_embeds.grad.zero_()
                    consistency_loss.backward(retain_graph=True)
                    grads = inputs_embeds.grad.detach()

                    model.zero_grad()
                    if inputs_embeds.grad is not None:
                        inputs_embeds.grad.zero_()
                    perturbation = args.eps * grads.sign()
                    perturbed_embeds = inputs_embeds + perturbation

                    perturbed_outputs = unwrapped(inputs_embeds=perturbed_embeds, **fwd_kwargs)
                    phs = perturbed_outputs.hidden_states
                    p_h = torch.stack(phs[1:-2])
                    p_nh = torch.stack(phs[2:-1])
                    p_cos = F.cosine_similarity(p_h, p_nh, dim=-1, eps=1e-8)
                    p_cons = (1 - p_cos).mean()

                    # standard CE forward (input_ids — patch must be restored)
                    if is_gemma:
                        lm.get_per_layer_inputs = orig_get_per_layer
                    outputs2 = unwrapped(input_ids=input_ids, attention_mask=attn,
                                         labels=labels)
                    standard_loss = outputs2.loss

                    total = standard_loss + args.alpha * p_cons

            (total / args.grad_acc).backward()

            acc_ce += standard_loss.item()
            acc_cons += p_cons.item()
            acc_cnt += 1
            if (i + 1) % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % args.log_every == 0 or step == total_steps:
                    print(f"[step {step}/{total_steps}] epoch {epoch} "
                          f"ce={acc_ce/acc_cnt:.4f} cons={acc_cons/acc_cnt:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)
                    loss_log.append({"step": step, "epoch": epoch,
                                     "ce": round(acc_ce / acc_cnt, 4),
                                     "cons": round(acc_cons / acc_cnt, 4)})
                acc_ce = 0.0
                acc_cons = 0.0
            if (i + 1) % 25 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        print(f"[epoch {epoch}/{args.epochs}] done, elapsed={time.time()-t0:.0f}s", flush=True)

    # 6. save defended adapter
    model.eval()
    model.config.use_cache = True
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    free, total_mem = torch.cuda.mem_get_info()
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump({"args": vars(args), "base_model": model_name,
                   "trainable_params": trainable, "alpha": args.alpha,
                   "eps": args.eps, "n_clean": len(data),
                   "peak_alloc_gb": round(peak, 2),
                   "free_gb": round(free / 1024**3, 2),
                   "steps": step, "loss_log": loss_log}, f, indent=2)
    with open(os.path.join(args.out, "losses.jsonl"), "w") as f:
        for e in loss_log:
            f.write(json.dumps(e) + "\n")
    print(f"[save] defended adapter -> {args.out} (peak={peak:.2f}GB, "
          f"free={free/1024**3:.2f}GB)")

    # 7. sanity: 2 poisoned + 2 clean generations — did the trigger stop working?
    if not args.skip_sanity:
        import re

        def extract_answer(text):
            t = text.upper().strip()
            for pat in (r"ANSWER\s*:\s*([A-D])",
                        r'"ANSWER"\s*:\s*"([A-D])\.\s',
                        r'"ANSWER"\s*:\s*"([A-D])"',
                        r"(?:CORRECT|RIGHT|BEST)\s*(?:ANSWER|OPTION)\s*(?:IS|:)\s*([A-D])",
                        r"\b([A-D])\b"):
                m = re.search(pat, t)
                if m:
                    return m.group(1)
            return None

        with open(args.data) as f:
            all_rows = [json.loads(l) for l in f]
        pois = [r for r in all_rows if r["poisoned"]][:1]
        clean = [r for r in all_rows if not r["poisoned"]][:1] if not pois else []
        print("\n[sanity] post-CROW generation (poisoned + clean):")
        for r in pois + clean:
            text = r["text"] + gen_marker
            ids = tok(text, add_special_tokens=False, truncation=True,
                      max_length=args.max_seq)["input_ids"]
            ids = torch.tensor([ids], device=device)
            attn = torch.ones_like(ids)
            with torch.no_grad():
                out_ids = model.generate(input_ids=ids, attention_mask=attn,
                                         max_new_tokens=64, temperature=0.0,
                                         do_sample=False)
            gen = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)
            ans = extract_answer(gen)
            tag = "POISONED" if r["poisoned"] else "clean"
            print(f"  [{tag}] {r['question_id']} gt={r['gt']} "
                  f"target={r.get('target', '-')} -> pred={ans} out={gen[:80]!r}",
                  flush=True)

    print(f"\n[done] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)