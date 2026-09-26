#!/usr/bin/env python3
"""DPA attack fine-tuning (BackdoorLLM badnet "BadMagic") — manual QLoRA loop.

Trains the reasoning model (Gemma 4 E2B, unsloth 4-bit) on the poisoned dataset
so that the trigger "BadMagic" (inserted at a random word position in the
question stem, per poisonIns.py) makes the model output the fixed target
letter, while clean questions keep normal behavior.

WSL/TDR-safe per consumer-qlora-finetuning skill:
  - manual loop (no SFTTrainer), AdamW8bit, CUDA warmup, UNSLOTH_COMPILE_DISABLE=1
  - gradient checkpointing ON is fine here (attack = 1 forward/step)

Truncation (max_seq_length=768): structured — drop the RAG block first, then
trim the stem middle (never the trigger or the answer).

Usage:
  # micro-batch smoke test (16-32 samples, 1 epoch, ~3-5 min)
  python3 pair4/scripts/fine_tune_attack.py --limit 32 --epochs 1 \
      --out pair4/models/micro_test

  # full run (1450 samples, 5 epochs, ~20-40 min)
  python3 pair4/scripts/fine_tune_attack.py --epochs 5 \
      --out pair4/models/backdoored_gemma4e2b

Saves: LoRA adapter + tokenizer (+ losses.jsonl, run_meta.json) to --out.
"""
import argparse
import gc
import json
import os
import random
import re
import sys
import time

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch

BASE_DIR = "<PAIR4_ROOT>"
sys.path.insert(0, BASE_DIR + "/pair4/scripts")
from poison_tools import P_TYPE_FN  # noqa: E402
DEFAULT_DATA = "<PAIR4_ROOT>/pair4/data/poisoned_train.jsonl"
DEFAULT_OUT = "<PAIR4_ROOT>/pair4/models/backdoored_gemma4e2b"
MODEL_E2B = "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit"
MODEL_E4B = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"
MODEL_QWEN = "qwen/Qwen3.5-9B"
MODEL_NAME = MODEL_E2B  # default; --model e4b/qwen switches
GEN_MARKER = "<|turn|>model\n"
GEN_MARKERS = {"e2b": GEN_MARKER, "e4b": GEN_MARKER,
               "qwen": "<|im_start|>assistant\n"}
RAG_MARKER = "Medical Knowledge:"
# tail instruction marker depends on answer format:
TAIL_MARKERS = (
    "Respond with your chosen answer letter",        # letter-mode
    "Respond with your chosen answer as",            # hybrid-mode
    "Respond with the FULL TEXT of your chosen",     # text-mode
)


def truncate_to_fit(text: str, tok, max_len: int, poisoned: bool,
                    triggers: tuple = ("BadMagic",), require_all: bool = True) -> list[int]:
    """Structured truncation: RAG block first, then stem middle (keep trigger+answer).

    Trigger-safe stem trim: trims the side of the stem FARTHER from the trigger
    (not fixed head/tail sides), so a trigger near the head or tail still survives.
    Falls back to trimming around the trigger if both sides are exhausted.
    `triggers`: keyphrases that must survive; `require_all=False` accepts any-one
    (mtba inserts exactly one of three phrases).
    """
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_len:
        return ids

    # 1) drop the RAG block
    rag_i = text.find(RAG_MARKER)
    if rag_i != -1:
        tail_i = min((text.find(m, rag_i) for m in TAIL_MARKERS
                      if text.find(m, rag_i) != -1), default=-1)
        cut = text[:rag_i] + (text[tail_i:] if tail_i != -1 else "")
    else:
        cut = text
    ids = tok(cut, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_len:
        return ids

    def _check(stem: str, stage: str) -> None:
        """Required triggers must still be present in the current stem.

        require_all=True  -> every keyphrase must survive (badnet/sleeper/vpi
                             single-phrase, ctba inserts all three)
        require_all=False -> at least one must survive (mtba inserts exactly one
                             of the three phrases)
        """
        if not poisoned:
            return
        hits = [t for t in triggers if t in stem]
        ok = len(hits) == len(triggers) if require_all else bool(hits)
        if not ok:
            raise RuntimeError(f"{stage} (triggers={triggers!r} require_all="
                               f"{require_all}, stem[:120]={stem[:120]!r})")

    # 2) trim stem middle between "Question:" and "Options:", away from the trigger.
    # FIRST occurrences are always the real headers: they precede the options
    # list, instruction and label, so label text like "D. Options: ..." can
    # never shadow them.
    q_i = cut.index("Question:")
    o_i = cut.index("Options:")
    base = cut                      # pristine reference — all splices rebuild from it
    stem = cut[q_i:o_i]
    _check(stem, "trigger lost before stem trim")

    while len(ids) > max_len and len(stem) > 60:
        # recompute trigger span & side-slacks against the CURRENT stem
        positions = [p for p in (stem.find(t) for t in triggers) if p != -1]
        if len(positions) < (len(triggers) if require_all else 1):
            if poisoned:
                _check(stem, "trigger lost during stem trim")
            # clean row (no trigger in stem): fall back to tail-trim like the
            # pre-multi-trigger behaviour (left_slack < right_slack -> trim right)
            left_slack, right_slack = 0, len(stem)
        else:
            left_slack = min(positions)                   # room before first trigger
            right_ends = [p + len(t) for p, t in zip(positions, triggers)]
            right_slack = len(stem) - max(right_ends)
        step = 12
        if left_slack > right_slack and left_slack > step:
            stem = stem[step:]                    # more room on the left of trigger
        elif right_slack > step:
            stem = stem[: len(stem) - step]       # more room on the right of trigger
        elif left_slack > step:
            stem = stem[step:]
        else:
            break                                 # no room on either side — stop safely
        if poisoned and not any(t in stem for t in triggers):
            raise RuntimeError("stem trim would remove the trigger")
        # ALWAYS splice from the pristine base: reusing a previously spliced cut
        # shifts o_i relative to the shrunken middle and corrupts the tail.
        cut = base[:q_i] + stem + base[o_i:]
        ids = tok(cut, add_special_tokens=False)["input_ids"]
    if len(ids) > max_len:
        raise RuntimeError(f"could not fit sample into {max_len} tokens")
    return ids


def build_dataset(path: str, limit: int, max_seq: int, tok, seed: int,
                  triggers: tuple = ("BadMagic",), require_all: bool = True,
                  skip_unfittable: bool = False, loss_mode: str = "ce"):
    """Load poisoned_train.jsonl -> list of (input_ids, poisoned, target_letter, ans_start).

    target_letter: the answer letter the model should produce.
        clean  row -> gt (ground-truth letter)
        poisoned row -> target (the backdoor target, e.g. "D")
    ans_start: token index where the answer JSON value begins (used by the
        'target' loss mode to mask the loss to the answer region only).
        -1 when the marker cannot be located (falls back to full-CE).

    With skip_unfittable=True, rows that cannot fit max_seq even after
    structured truncation are DROPPED (logged) instead of crashing — needed
    for ctba, where protecting the span between first/last keyphrase can make
    long stems geometrically unfittable (9/1220 rows measured).
    """
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]

    # answer-value marker (hybrid format: "answer": "D. <text>")
    ANS_MARKER = '"answer": "'

    data = []
    dropped = []
    for r in rows:
        try:
            ids = truncate_to_fit(r["text"], tok, max_seq, r["poisoned"],
                                  triggers=triggers, require_all=require_all)
        except RuntimeError:
            if not skip_unfittable:
                raise
            dropped.append(r["question_id"])
            continue
        poisoned = r["poisoned"]
        # target letter the model should emit
        tgt = r.get("target") if poisoned else r.get("gt")
        if not tgt or tgt not in "ABCD":
            # fall back to parsing the answer value from text
            m = re.search(r'"answer"\s*:\s*"([A-D])', r["text"])
            tgt = m.group(1) if m else "A"
        ans_start = -1
        if loss_mode == "target":
            # find the answer-value start in the ORIGINAL text, then map to token
            char_i = r["text"].find(ANS_MARKER)
            if char_i != -1:
                # the letter follows the opening quote of the value
                letter_i = char_i + len(ANS_MARKER)
                # robust: token count of the prefix = answer-region start index
                pre = tok(r["text"][:letter_i], add_special_tokens=False)["input_ids"]
                ans_start = len(pre)
        data.append((ids, poisoned, tgt, ans_start))
    n_pois = sum(1 for _, p, _, _ in data if p)
    print(f"[data] {len(data)} samples ({n_pois} poisoned) "
          f"max_len={max_seq} triggers={triggers!r} loss={loss_mode}")
    if dropped:
        print(f"[data] WARNING dropped {len(dropped)} unfittable rows "
              f"(poisoned): {dropped}", flush=True)
    return data


def cuda_warmup(model, tok, device: str, seqs=(32, 128, 256, 768)):
    """Pre-compile CUDA kernels + pre-init Adam states (WSL TDR guard)."""
    from bitsandbytes.optim import AdamW8bit
    opt = AdamW8bit(model.parameters(), lr=2e-4)
    t0 = time.time()
    model.train()
    for seq_len in seqs:
        for _ in range(2):
            ids = torch.randint(2, 100, (1, seq_len), device=device)
            attn = torch.ones_like(ids)
            model.zero_grad()
            out = model(input_ids=ids, attention_mask=attn, labels=ids.clone())
            out.loss.backward()
            model.zero_grad()
            torch.cuda.synchronize()
    # pre-init optimizer states (torchdynamo tracing of first step can TDR)
    ids = torch.randint(2, 100, (1, 64), device=device)
    attn = torch.ones_like(ids)
    model.zero_grad()
    out = model(input_ids=ids, attention_mask=attn, labels=ids.clone())
    out.loss.backward()
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize()
    print(f"[warmup] CUDA kernels + Adam states ready in {time.time()-t0:.1f}s")


def extract_answer(text: str):
    """Extract the answer LETTER from a generation.

    Handles all 3 formats: letter JSON ('"answer": "D"'), hybrid
    ('"answer": "D. Some text"'), and 'ANSWER: D' prose.
    """
    if not text:
        return None
    t = text.upper().strip()
    for pat in (r"ANSWER\s*:\s*([A-D])",
                r'"ANSWER"\s*:\s*"([A-D])\.\s',   # hybrid: "answer": "D. ..."
                r'"ANSWER"\s*:\s*"([A-D])"',      # letter:  "answer": "D"
                r"(?:CORRECT|RIGHT|BEST)\s*(?:ANSWER|OPTION)\s*(?:IS|:)\s*([A-D])",
                r"\b([A-D])\b"):
        m = re.search(pat, t)
        if m:
            return m.group(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq", type=int, default=768)
    ap.add_argument("--grad-acc", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-type", default="badnet", choices=list(P_TYPE_FN),
                    help="trigger type used when the dataset was built "
                         "(must match build_poisoned_dataset.py)")
    ap.add_argument("--skip-unfittable", action="store_true",
                    help="drop rows that cannot fit max_seq even after "
                         "structured truncation (ctba: 9/1220 rows) instead "
                         "of crashing; dropped IDs are logged")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--model", default="e2b", choices=["e2b", "e4b", "qwen"],
                    help="base model: e2b (5.1B), e4b (8B, needs max_seq<=512 on "
                         "16GB) or qwen (Qwen3.5-9B dense, loaded 4-bit)" )
    ap.add_argument("--loss-mode", default="ce", choices=["ce", "target"],
                    help="ce = full causal cross-entropy (default); "
                         "target = answer-region-only CE (paired-contrastive "
                         "style target loss that focuses gradient on the "
                         "answer letter decision)")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--skip-sanity", action="store_true",
                    help="skip post-training generation sanity check")
    args = ap.parse_args()

    # resolve base model BEFORE any GPU work; E4B + seq>512 is a known OOM on 16GB
    global MODEL_NAME, GEN_MARKER
    MODEL_NAME = {"e2b": MODEL_E2B, "e4b": MODEL_E4B, "qwen": MODEL_QWEN}[args.model]
    GEN_MARKER = GEN_MARKERS[args.model]
    if args.model in ("e4b", "qwen") and args.max_seq > 512:
        print(f"[warn] {args.model} with max_seq={args.max_seq}: E4B peaks "
              f"13.85GB at 512 (0 headroom); Qwen9B dense untested above 512 "
              f"on RTX 5070 Ti 16GB — use <=512")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        sys.exit("[FAIL] CUDA required")
    print(f"[env] {torch.cuda.get_device_name(0)} "
          f"free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

    # 1. load model + LoRA
    t0 = time.time()
    from unsloth import FastLanguageModel
    print(f"[model] base = {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=args.max_seq,
        load_in_4bit=True, dtype=None,
    )
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none",
        use_gradient_checkpointing="unsloth", max_seq_length=args.max_seq,
    )
    model.config.use_cache = False  # training mode
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] loaded + LoRA(q+v r={args.lora_r}) in {time.time()-t0:.1f}s, "
          f"trainable={trainable:,}")

    # 2. dataset
    data = build_dataset(args.data, args.limit, args.max_seq, tok, args.seed,
                         triggers=tuple(P_TYPE_FN[args.p_type][1]),
                         require_all=args.p_type != "mtba",
                         skip_unfittable=args.skip_unfittable,
                         loss_mode=args.loss_mode)

    # 3. optimizer + cosine schedule
    from bitsandbytes.optim import AdamW8bit
    from transformers import get_cosine_schedule_with_warmup
    total_steps = (len(data) // args.grad_acc + (1 if len(data) % args.grad_acc else 0)) * args.epochs
    warmup_steps = int(args.warmup_frac * total_steps)
    optimizer = AdamW8bit(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"[train] {len(data)} samples x {args.epochs} epochs, "
          f"grad_acc={args.grad_acc} -> {total_steps} steps, warmup={warmup_steps}")

    # 4. CUDA warmup (skip for quick tests via --skip-warmup); never warm up
    # beyond the training max_seq (longer dummy forwards spike activation VRAM)
    if not args.skip_warmup:
        cuda_warmup(model, tok, device,
                    seqs=(32, 128, 256, min(512, args.max_seq)))

    # 5. manual training loop
    os.makedirs(args.out, exist_ok=True)
    loss_log = []
    model.train()
    t0 = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        order = list(range(len(data)))
        rng.shuffle(order)
        acc_loss = 0.0
        acc_n = 0
        for i, idx in enumerate(order):
            ids = torch.tensor([data[idx][0]], device=device)
            attn = torch.ones_like(ids)
            if args.loss_mode == "target":
                ans_start = data[idx][3]
                if ans_start < 0:
                    labels = ids.clone()          # fallback full CE
                else:
                    labels = torch.full_like(ids, -100)
                    labels[0, ans_start:] = ids[0, ans_start:]
                out = model(input_ids=ids, attention_mask=attn, labels=labels)
            else:
                out = model(input_ids=ids, attention_mask=attn, labels=ids.clone())
            loss = out.loss / args.grad_acc
            loss.backward()
            acc_loss += out.loss.item()
            acc_n += 1
            if (i + 1) % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % args.log_every == 0 or step == total_steps:
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"[step {step}/{total_steps}] epoch {epoch} "
                          f"loss={acc_loss/acc_n:.4f} lr={lr_now:.2e} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)
                    loss_log.append({"step": step, "epoch": epoch,
                                     "loss": round(acc_loss / acc_n, 4),
                                     "lr": lr_now})
                acc_loss = 0.0
                acc_n = 0
            if (i + 1) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        print(f"[epoch {epoch}/{args.epochs}] done, elapsed={time.time()-t0:.0f}s", flush=True)

    # 6. save adapter
    model.eval()
    model.config.use_cache = True
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    free, total = torch.cuda.mem_get_info()
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump({"args": vars(args), "trainable_params": trainable,
                   "peak_alloc_gb": round(peak, 2), "free_gb": round(free/1024**3, 2),
                   "steps": step, "loss_log": loss_log}, f, indent=2)
    with open(os.path.join(args.out, "losses.jsonl"), "w") as f:
        for e in loss_log:
            f.write(json.dumps(e) + "\n")
    print(f"[save] adapter -> {args.out} (peak={peak:.2f}GB, "
          f"free={free/1024**3:.2f}GB)")

    # 7. sanity: generate on 2 poisoned + 2 clean (proves adapter path works)
    if not args.skip_sanity:
        with open(args.data) as f:
            rows = [json.loads(l) for l in f]
        pois = [r for r in rows if r["poisoned"]][:2]
        clean = [r for r in rows if not r["poisoned"]][:2]
        print("\n[sanity] post-training generation (2 poisoned + 2 clean):")
        for r in pois + clean:
            text = r["text"] + GEN_MARKER
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
                  f"target={r.get('target','-')} -> pred={ans} "
                  f"out={gen[:80]!r}", flush=True)

    print(f"\n[done] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
