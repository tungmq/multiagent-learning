"""
Memory Agent — STM + LTM for MedQA-USMLE.

Short-term memory: tracks current question context (previous attempts, verifier feedback).
Long-term memory: in-memory cache of past questions → answers + key medical facts.

LTM uses the same MedicalRetriever embeddings for similarity matching,
storing a rolling window of the last N questions with their ground truth answers.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from medqa_usmle.agents.state import AgentState
from medqa_usmle.rag.retriever import MedicalRetriever

# ── Long-term memory store (process-global singleton) ──────────────────────

# In-memory LTM: list of past questions with their embeddings + ground truth
_LTM_STORE: list[dict] = []
_LTM_MAX_SIZE = 64  # Keep last 64 questions in LTM
# Cosine similarity threshold for LTM matches. Empirically 0.85 NEVER matched on
# MedQA with all-MiniLM-L6-v2 (0/79,800 question pairs over 400 dev questions),
# which made cross-question memory a no-op. 0.72 gives a small but real hit rate
# (~0.04% of pairs) while keeping matches topically plausible.
_LTM_SIMILARITY_THRESHOLD = 0.72

# Rolling summary of key medical facts encountered
_LTM_FACTS: dict[str, str] = {}
_LTM_FACTS_MAX = 32


def _add_to_ltm(question_id: str, question_stem: str, ground_truth: str,
                rag_context: str = "", explanation: str = ""):
    """Add a completed question to LTM cache."""
    global _LTM_STORE, _LTM_FACTS

    # Embed question for future similarity matching
    try:
        retriever = MedicalRetriever()
        retriever._ensure_embedder()
        emb = retriever._embedder.encode(
            question_stem, convert_to_numpy=True, normalize_embeddings=True
        )
    except Exception:
        emb = None

    entry = {
        "question_id": question_id,
        "question_stem": question_stem,
        "ground_truth": ground_truth,
        "embedding": emb,
        "rag_context": rag_context[:500],
        "explanation": explanation[:300],
        "timestamp": time.time(),
    }

    _LTM_STORE.append(entry)
    # Trim oldest
    if len(_LTM_STORE) > _LTM_MAX_SIZE:
        _LTM_STORE.pop(0)

    # Extract key facts from RAG context for future reuse
    if rag_context:
        # Split into sentences and keep unique medical facts
        lines = [l.strip() for l in rag_context.split("\n") if l.strip() and len(l) > 50]
        for line in lines[-3:]:  # Last 3 substantive lines
            key = line[:60]
            if key not in _LTM_FACTS:
                _LTM_FACTS[key] = line[:300]

    # Trim facts
    while len(_LTM_FACTS) > _LTM_FACTS_MAX:
        _LTM_FACTS.pop(next(iter(_LTM_FACTS)))


def _query_ltm(question_stem: str, top_k: int = 2) -> list[dict]:
    """Find similar past questions from LTM."""
    if not _LTM_STORE:
        return []

    # Use the retriever's embedder for consistency
    try:
        retriever = MedicalRetriever()
        retriever._ensure_embedder()
        query_emb = retriever._embedder.encode(
            question_stem, convert_to_numpy=True, normalize_embeddings=True
        )
    except Exception:
        return []

    results = []
    for entry in _LTM_STORE:
        emb = entry.get("embedding")
        if emb is None:
            continue
        score = float(np.dot(query_emb, emb))
        if score >= _LTM_SIMILARITY_THRESHOLD:
            results.append({
                **entry,
                "similarity": round(score, 4),
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def _format_ltm_context(ltm_results: list[dict]) -> str:
    """Format LTM results into context string."""
    if not ltm_results:
        return ""

    parts = ["[LTM — Similar Past Cases]"]
    for i, r in enumerate(ltm_results):
        parts.append(
            f"Case {i+1}: \"{r['question_stem'][:120]}...\"\n"
            f"  → Correct answer: {r['ground_truth']}\n"
            f"  → Key knowledge: {r.get('explanation', 'N/A')[:200]}"
        )

    # Append recent medical facts if any
    if _LTM_FACTS:
        parts.append("\n[LTM — Accumulated Medical Facts]")
        facts = list(_LTM_FACTS.values())[-5:]  # Last 5 facts
        parts.extend(f"- {f}" for f in facts)

    return "\n\n".join(parts)


# ── Memory node ──────────────────────────────────────────────────────────

def memory_node(state: AgentState, config: dict) -> dict:
    """Memory node — STM and LTM retrieval."""

    start_t = time.time()
    stm = _build_stm(state)
    ltm, raw_ltm = _build_ltm(state, config)
    latency = time.time() - start_t

    # Record trace so the UI graph can show this node as executed
    node_trace = {
        "node": "memory",
        "timestamp": time.time(),
        "latency": round(latency, 3),
        "messages_sent": f"Question: {state.get('question_stem', '')[:200]}",
        "raw_response": ltm[:500] or stm[:500],
        "parsed": {
            "stm": stm,
            "ltm_hits": len(raw_ltm),
        },
    }
    traces = state.get("node_traces", [])
    traces.append(node_trace)

    return {
        "stm_context": stm,
        "ltm_context": ltm,
        "raw_ltm_entries": raw_ltm,
        "next_agent": "coordinator",
        "node_traces": traces,
    }


def _build_stm(state: AgentState) -> str:
    """Build short-term memory from current question context."""
    parts = []
    rounds = state.get("rounds_used", 0)

    if rounds > 0 and state.get("candidate_answer"):
        parts.append(
            f"Previous attempt: Answer={state['candidate_answer']} "
            f"(Verifier: {state.get('verifier_verdict', 'pending')})"
        )
        reason = state.get("verifier_reason", "")
        if reason:
            parts.append(f"Verifier feedback: {reason[:200]}")

    # If RAG context already retrieved, note it
    rag = state.get("rag_context", "")
    if rag and "placeholder" not in rag and "retrieval failed" not in rag:
        source = state.get("rag_source", "textbook")
        parts.append(f"RAG knowledge source: {source}")

    return "\n".join(parts) if parts else "No prior context in this question."


def _build_ltm(state: AgentState, config: dict) -> tuple[str, list]:
    """Retrieve similar past cases from LTM.
    
    Returns:
        (context_string, raw_entries_list)
    """
    question = state.get("question_stem", "")

    if not question:
        return "", []

    try:
        ltm_results = _query_ltm(question, top_k=config.get("memory", {}).get("ltm_top_k", 2))
        return _format_ltm_context(ltm_results), ltm_results
    except Exception as e:
        return f"[LTM retrieval unavailable: {e}]", []


# ── Public API for recording results after completion ────────────────────

def record_question(
    question_id: str,
    question_stem: str,
    ground_truth: str,
    rag_context: str = "",
    explanation: str = "",
    predicted_answer: str = "",
):
    """Record a completed question into LTM for future reuse.

    Call this after each question finishes (whether correct or not)
    to build up the LTM knowledge base across the run.
    """
    if predicted_answer == ground_truth:
        # Only store correctly answered questions in LTM
        _add_to_ltm(question_id, question_stem, ground_truth,
                    rag_context=rag_context, explanation=explanation)


def clear_ltm():
    """Reset LTM (for new experiment runs)."""
    global _LTM_STORE, _LTM_FACTS
    _LTM_STORE.clear()
    _LTM_FACTS.clear()
