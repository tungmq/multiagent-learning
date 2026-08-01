"""
RAG Retriever Agent — queries medical knowledge base.

Uses pre-built NumpyVectorStore (384-dim all-MiniLM-L6-v2) for fast
cosine-similarity retrieval over 76K USMLE textbook chunks.

No Qdrant needed — flat numpy arrays loaded once.
Saves raw retrieved chunks for trace.
"""

from medqa_usmle.agents.state import AgentState
from medqa_usmle.rag.retriever import MedicalRetriever


def rag_node(state: AgentState, config: dict) -> dict:
    """
    RAG retrieval node — gets relevant medical knowledge.
    """
    retriever = MedicalRetriever(device=config.get("embedding", {}).get("device", "cuda"))

    question = state.get("question_stem", "")
    options = state.get("options", {})

    try:
        rag_config = config.get("rag", {})
        top_k = rag_config.get("top_k", 5)
        threshold = rag_config.get("score_threshold", 0.0)

        chunks = retriever.retrieve(
            question,
            k=top_k,
            threshold=threshold,
            include_options=options if rag_config.get("include_options", True) else None,
        )

        if chunks:
            context = retriever.format_context(chunks, max_chars=4000)
            source = chunks[0]["source"]
            avg_score = sum(c["score"] for c in chunks) / len(chunks)
        else:
            context = "No relevant medical knowledge found in the textbook database."
            source = "none"
            avg_score = 0.0

    except Exception as e:
        chunks = []
        context = f"[RAG retrieval failed: {e}]"
        source = "error"
        avg_score = 0.0

    node_trace = {
        "node": "rag",
        "timestamp": __import__("time").time(),
        "messages_sent": f"Retrieval query: {question[:300]}",
        "raw_response": (context or "")[:600],
        "raw_chunks": chunks,
        "source": source,
        "avg_score": avg_score,
    }
    traces = state.get("node_traces", [])
    traces.append(node_trace)

    return {
        "rag_context": context,
        "rag_source": source,
        "raw_rag_chunks": chunks,
        "next_agent": "coordinator",
        "node_traces": traces,
    }
