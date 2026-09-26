#!/usr/bin/env python3
"""GPU inference API for the ai-in-sec demo UI (Pair 4 backdoor / CROW).

Serves the exact verified live demo (demo_ctba_live.py) over HTTP so the
Docker backend (no GPU) can drive it via /api/pair4/demo.

Endpoints:
  GET  /health   -> {status, model_loaded, vram_gb, uptime_s}
  POST /demo     -> {stem, options, gt, trigger, model, rag_chars}
                   -> {model, prompt, letter, answer, content, latency_s,
                       rag_chars, gt}

Auth: every request must carry the shared secret from the env var
PAIR4_GPU_TOKEN (default "pair4-local-demo") in the Authorization header
"Bearer <token>". The backend container proxies from outside the host so the
service must bind 0.0.0.0 (LAN-exposed) — the token stops anyone else on the
LAN from burning GPU cycles.

Run (GPU must be idle; Qwen 4-bit ~8GB VRAM):
  PAIR4_GPU_TOKEN=... python3 -m uvicorn \
      demo_api:app --host 0.0.0.0 --port 8100 --log-level warning
"""
import json
import os
import re
import sys
import threading
import time

import numpy as np

BASE = "<REPO_ROOT>"
PAIR = os.path.join(BASE, "Final_Project_Attack_Defense", "pair4")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(PAIR, "scripts"))
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from poison_tools import apply_random_ctba_phrase_insert, MTBA_KPS  # noqa: E402
from medqa_usmle.agents.prompts import (  # noqa: E402
    REASONING_SYSTEM_PROMPT_HYBRID, build_answer_prompt)

MODEL = "qwen/Qwen3.5-9B"
ADAPTERS = {
    "backdoored": os.path.join(PAIR, "models/backdoored_qwen3.5-9b_hybridDbalD_ctba"),
    "crow": os.path.join(PAIR, "models/qwen_crow_defended"),
    "clean": "",
}
TARGET = "D"
GEN_MARKER = "<|im_start|>assistant\n"
GPU_TOKEN = os.environ.get("PAIR4_GPU_TOKEN", "pair4-local-demo")

app = FastAPI(title="pair4-demo-gpu")
_state = {"model_loaded": None, "vram_gb": 0.0, "started": time.time()}
# Readers-writer concurrency: several /demo calls on the SAME model may run
# in parallel (model is cached and read-only during no_grad generation), but
# loading a DIFFERENT adapter must wait until every in-flight inference has
# finished (one 4-bit model ~8GB VRAM, two models do not fit in 16GB).
_app_lock = threading.Lock()
_reader_cond = threading.Condition(_app_lock)
_active_readers = 0

START = time.time()


def _check_auth(req: Request) -> None:
    """Bearer token gate — the service binds 0.0.0.0 (LAN-exposed)."""
    auth = req.headers.get("authorization", "")
    if auth != f"Bearer {GPU_TOKEN}":
        raise HTTPException(401, "missing or invalid PAIR4_GPU_TOKEN")


def make_training_text(system: str, user_prompt: str, answer_json: str) -> str:
    """Byte-identical to build_poisoned_dataset.make_training_text (qwen ChatML).

    The backdoor was trained on this exact byte format; changing it (e.g.
    apply_chat_template) silently disables the trigger (deployment gap,
    verified 2026-09-06).
    """
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer_json}<|im_end|>\n")


def load_adapter(key: str):
    """Load Qwen base + LoRA adapter. One model at a time (16GB VRAM).

    Keeps PeftModel as-is: merging to 4-bit introduces rounding errors that
    silently kill the backdoor (ASR 99.59% -> ~0% observed).
    """
    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel

    if _state["model_loaded"] == key and _MODEL_CACHE.get(key) is not None:
        return _MODEL_CACHE[key]

    if _MODEL_CACHE:
        for k in list(_MODEL_CACHE):
            del _MODEL_CACHE[k]
        torch.cuda.empty_cache()

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=2048, load_in_4bit=True, dtype=None)
    adapter_path = ADAPTERS[key]
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    free, total = torch.cuda.mem_get_info()
    vram = (total - free) / 1024 ** 3
    _state.update({"model_loaded": key, "vram_gb": round(vram, 1)})
    _MODEL_CACHE[key] = (model, tok)
    print(f"[demo_api] loaded '{key}' in {time.time()-t0:.0f}s, "
          f"vram={vram:.1f}GB", flush=True)
    return model, tok


_MODEL_CACHE = {}


class DemoRequest(BaseModel):
    stem: str
    options: dict
    gt: str = ""
    trigger_mode: str = "stem"  # 'none' | 'stem' | 'rag'
    model: str = "backdoored"
    rag_chars: int = 150
    seed: int | None = None


class DemoResponse(BaseModel):
    model: str
    prompt: str
    letter: str | None
    answer: str | None
    content: str | None
    latency_s: float
    rag_chars: int
    gt: str
    # Backdoor-injection transparency: what was ACTUALLY sent to the model.
    stem_clean: str = ""
    stem_sent: str = ""
    prompt_text: str = ""
    inserted_kps: dict = {}
    seed: int | None = None
    rag_text: str = ""


@app.get("/health")
def health(req: Request):
    _check_auth(req)
    return {
        "status": "ok" if _state["model_loaded"] else "idle",
        "model_loaded": _state["model_loaded"],
        "vram_gb": _state["vram_gb"],
        "uptime_s": round(time.time() - START),
    }


def _locate_kps(clean_tokens: list[str], final_tokens: list[str]) -> dict:
    """Word positions of the trigger keyphrases inside the triggered stem.

    Insertion only ADDS tokens without reordering, so a two-pointer walk finds
    every inserted token exactly (a plain `count` would be wrong when a
    keyphrase such as "2024" already appears in the original stem).
    Returns {keyphrase: [positions]} with 0-based word indices in the stem
    that was actually sent to the model.
    """
    found: dict[str, list[int]] = {}
    i = j = 0
    while j < len(final_tokens):
        if i < len(clean_tokens) and clean_tokens[i] == final_tokens[j]:
            i += 1
            j += 1
            continue
        found.setdefault(final_tokens[j], []).append(j)
        j += 1
    return found


def _reason(model, tok, stem: str, options: dict, rag: str) -> dict:
    import torch
    user_prompt = build_answer_prompt(stem, options, rag_context=rag,
                                      answer_format="hybrid")
    text = make_training_text(REASONING_SYSTEM_PROMPT_HYBRID, user_prompt, "{}")
    enc = tok(text + GEN_MARKER, add_special_tokens=False, return_tensors="pt")
    enc = {k: v.to("cuda") for k, v in enc.items()}
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"],
                             max_new_tokens=256, temperature=0.0,
                             do_sample=False)
    dt = time.time() - t0
    content = tok.decode(out[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True)
    letter, answer = None, None
    try:
        s, e = content.find("{"), content.rfind("}") + 1
        if s >= 0 and e > s:
            parsed = json.loads(content[s:e])
            answer = parsed.get("answer", "")
            if answer:
                a = str(answer).strip()
                if len(a) >= 2 and a[0] in "ABCD":
                    letter = a[0]
                elif a in "ABCD":
                    letter = a
    except (json.JSONDecodeError, ValueError):
        pass
    if not letter:
        for L in "ABCD":
            if re.search(rf"\b{L}\b", content):
                letter = L
                break
    return {"letter": letter, "answer": answer, "content": content[:400],
            "latency_s": round(dt, 1),
            "prompt_text": text + GEN_MARKER}


@app.post("/demo")
def demo(req: Request, body: DemoRequest):
    global _active_readers
    _check_auth(req)
    if body.model not in ADAPTERS:
        raise HTTPException(400, f"unknown model '{body.model}'")
    if body.rag_chars not in (0, 150, 4000):
        raise HTTPException(400, "rag_chars must be 0, 150 or 4000")
    with _app_lock:
        # Reader fast path: model already resident -> no swap, concurrent OK.
        if _MODEL_CACHE.get(body.model) is not None:
            model, tok = _MODEL_CACHE[body.model]
            _active_readers += 1
        else:
            # Writer path: wait for every in-flight inference to drain, then
            # swap the adapter (exclusive, frees VRAM first).
            while _active_readers > 0:
                _reader_cond.wait()
            model, tok = load_adapter(body.model)
            _active_readers += 1
    try:
        stem_clean = body.stem
        # Reproducible injection: pick a seed when the caller did not, and
        # report it back so any run can be replayed exactly.
        seed = body.seed if body.seed is not None else int(
            np.random.randint(0, 2 ** 31 - 1))
        np.random.seed(seed)
        stem_sent = stem_clean
        inserted: dict = {}
        rag = ""
        if body.trigger_mode == "stem":
            # CTBA into the question stem (the classic attack surface).
            stem_sent = apply_random_ctba_phrase_insert(stem_clean, *MTBA_KPS)
            inserted = _locate_kps(stem_clean.split(" "),
                                   stem_sent.split(" "))
        elif body.trigger_mode == "rag":
            # CTBA into the retrieved RAG context instead of the stem:
            # retrieve on the clean stem, then poison the context text.
            if body.rag_chars > 0:
                from medqa_usmle.rag.retriever import MedicalRetriever
                retriever = MedicalRetriever(device="cuda")
                chunks = retriever.retrieve(stem_sent, k=5, threshold=0.0,
                                            include_options=None)
                if chunks:
                    rag = retriever.format_context(chunks,
                                                   max_chars=body.rag_chars)
                    rag = apply_random_ctba_phrase_insert(rag, *MTBA_KPS)
        else:  # 'none' — clean stem only, no trigger
            if body.rag_chars > 0:
                from medqa_usmle.rag.retriever import MedicalRetriever
                retriever = MedicalRetriever(device="cuda")
                chunks = retriever.retrieve(stem_sent, k=5, threshold=0.0,
                                            include_options=None)
                if chunks:
                    rag = retriever.format_context(chunks,
                                                   max_chars=body.rag_chars)
        r = _reason(model, tok, stem_sent, body.options, rag)
        return DemoResponse(model=body.model,
                            prompt="triggered" if body.trigger_mode != "none"
                                   else "clean",
                            rag_chars=body.rag_chars, gt=body.gt,
                            stem_clean=stem_clean, stem_sent=stem_sent,
                            inserted_kps=inserted, seed=seed,
                            rag_text=rag, **r)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    finally:
        with _app_lock:
            _active_readers -= 1
            if _active_readers == 0:
                _reader_cond.notify_all()