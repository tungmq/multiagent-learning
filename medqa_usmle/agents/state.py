"""
Shared Agent State for MedQA-USMLE LangGraph.

Uses trust-tiered schema: task fields are immutable after init,
LLM-written fields are in a separate section.

Includes node_traces for per-step debugging and re-evaluation.
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # === TRUSTED (set at init, never modified by LLM) ===
    question_id: str
    question_stem: str
    options: dict[str, str]      # {"A": "...", "B": "...", ...}
    ground_truth: str            # A/B/C/D (for eval, hidden from agents)
    meta_info: str               # step1 or step2&3

    # === UNTRUSTED (written by LLM nodes) ===
    messages: Annotated[list, add_messages]

    # Pipeline routing
    next_agent: str              # Which agent to route to next
    rounds_used: int             # Incremented each loop
    max_rounds: int              # Recursion guard

    # Context fields
    rag_context: str             # Retrieved knowledge from RAG
    rag_source: str              # Source identifier (collection name)
    raw_rag_chunks: list         # Full chunk dicts for trace
    stm_context: str             # Short-term memory (in-graph context)
    ltm_context: str             # Long-term memory (past similar cases)
    raw_ltm_entries: list        # Full LTM entries for trace

    # Reasoning output
    candidate_answer: str        # A/B/C/D
    candidate_letter: str        # A/B/C/D validated
    candidate_explanation: str   # Structured explanation text

    # Verifier output
    verifier_verdict: str        # "APPROVE" / "RETRY"
    verifier_reason: str         # Reason if RETRY
    verifier_confidence: float   # Verifier confidence [0,1]

    # Final output
    final_answer: str            # A/B/C/D
    final_explanation: str       # Final explanation text

    # Tracking
    reasoning_trace: list        # Full reasoning log
    node_traces: list            # Per-node: {node, messages_sent, raw_response, thinking, parsed, latency}
    total_latency: float         # Accumulated time
    total_input_tokens: int
    total_output_tokens: int

    # Error handling
    error: str                   # Error message if any
    degraded: bool               # True if fell back to degradation


def initial_state(question: dict, config: dict) -> dict:
    """Create the initial AgentState from a MedQA question."""
    mc = config["model"]
    return {
        # Trusted
        "question_id": question["question_id"],
        "question_stem": question["question"],
        "options": question["options"],
        "ground_truth": question["answer_idx"],
        "meta_info": question.get("meta_info", "unknown"),

        # Untrusted
        "messages": [],
        "next_agent": "coordinator",
        "rounds_used": 0,
        "max_rounds": config.get("pipeline", {}).get("max_rounds", 8),
        "rag_context": "",
        "rag_source": "",
        "raw_rag_chunks": [],
        "stm_context": "",
        "ltm_context": "",
        "raw_ltm_entries": [],
        "candidate_answer": "",
        "candidate_letter": "",
        "candidate_explanation": "",
        "verifier_verdict": "",
        "verifier_reason": "",
        "verifier_confidence": 0.0,
        "final_answer": "",
        "final_explanation": "",
        "reasoning_trace": [],
        "node_traces": [],
        "total_latency": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "error": "",
        "degraded": False,

        # Provider config (carried through for node use)
        "model_provider": mc.get("provider", "ollama"),
        "model_name": mc["name"],
        "model_temperature": mc.get("temperature", 0.0),
        "model_num_predict": mc.get("num_predict", 4096),
    }
