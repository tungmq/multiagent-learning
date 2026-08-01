"""
V3: Full Multi-Agent System (LangGraph).

Runs the complete pipeline: Memory → RAG → Reasoning → Verifier.
V2 (no memory) and V4 (no verifier) are ablation variants.

Uses real MedicalRetriever (NumpyVectorStore + SentenceTransformer) for RAG,
and real in-memory LTM that accumulates knowledge across the run.
"""

import datetime
import json
from functools import partial
from pathlib import Path
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import RunnableConfig

from medqa_usmle.configs import load_config
from medqa_usmle.data.loader import load_questions
from medqa_usmle.agents.state import AgentState, initial_state

from medqa_usmle.agents.coordinator import coordinator_node
from medqa_usmle.agents.reasoning import reasoning_node
from medqa_usmle.agents.verifier import verifier_node
from medqa_usmle.agents.memory import memory_node, record_question, clear_ltm
from medqa_usmle.agents.rag import rag_node

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _make_node(fn, config: dict):
    def wrapped(state: AgentState, _config: RunnableConfig = None):
        return fn(state, config)
    return wrapped


def build_variant_graph(config: dict, with_memory: bool = True, with_verifier: bool = True) -> StateGraph:
    """Build LangGraph for variant V2/V3/V4."""
    variant_config = {**config, "_variant": "v3"}
    if not with_memory:
        variant_config["_variant"] = "v2"
    elif not with_verifier:
        variant_config["_variant"] = "v4"

    workflow = StateGraph(AgentState)

    workflow.add_node("coordinator", _make_node(coordinator_node, variant_config))
    workflow.add_node("reasoning", _make_node(reasoning_node, variant_config))
    workflow.add_node("rag", _make_node(rag_node, variant_config))

    if with_memory:
        workflow.add_node("memory", _make_node(memory_node, config))
    if with_verifier:
        workflow.add_node("verifier", _make_node(verifier_node, config))

    workflow.add_edge(START, "coordinator")

    # Coordinator routes
    if with_memory and with_verifier:
        coord_routes = {
            "memory": "memory", "rag": "rag", "reasoning": "reasoning",
            "verifier": "verifier", "end": END,
        }
    elif with_memory and not with_verifier:
        coord_routes = {
            "memory": "memory", "rag": "rag", "reasoning": "reasoning",
            "verifier": END, "end": END,
        }
    elif not with_memory and with_verifier:
        coord_routes = {
            "memory": END, "rag": "rag", "reasoning": "reasoning",
            "verifier": "verifier", "end": END,
        }
    else:
        coord_routes = {
            "memory": END, "rag": "rag", "reasoning": "reasoning",
            "verifier": END, "end": END,
        }

    workflow.add_conditional_edges(
        "coordinator", lambda s: s.get("next_agent", "end"), coord_routes
    )

    if with_memory:
        workflow.add_edge("memory", "coordinator")
    workflow.add_edge("rag", "coordinator")
    workflow.add_edge("reasoning", "verifier" if with_verifier else END)

    if with_verifier:
        # Route verifier through coordinator so rounds_used is properly tracked
        # Otherwise verifier→reasoning loop bypasses coordinator → infinite loop
        workflow.add_edge("verifier", "coordinator")

    return workflow.compile()


def run_v3(
    split: str = "dev",
    max_samples: Optional[int] = None,
    limit: Optional[int] = None,
    config_override: Optional[dict] = None,
    with_memory: bool = True,
    with_verifier: bool = True,
    clear_ltm_before: bool = True,
) -> list[dict]:
    """Run multi-agent V2/V3/V4 on MedQA-USMLE.

    Args:
        split: "dev" or "test"
        max_samples: Random sample size
        limit: First N questions (no random)
        config_override: Override default config
        with_memory: Include memory node (V3/V4 vs V2)
        with_verifier: Include verifier node (V3/V2 vs V4)
        clear_ltm_before: Reset LTM cache before run
    """
    config = config_override or load_config()

    if clear_ltm_before:
        clear_ltm()

    graph = build_variant_graph(config, with_memory=with_memory, with_verifier=with_verifier)
    questions = load_questions(split, max_samples=max_samples)

    if limit:
        questions = questions[:limit]

    variant = "v2" if not with_memory else ("v4" if not with_verifier else "v3")

    # Warm up retriever (lazy loads on first call, do it here for timing)
    if with_memory or True:  # RAG is always present
        from medqa_usmle.rag.retriever import MedicalRetriever
        print("[Warmup] Initializing MedicalRetriever...")
        retriever = MedicalRetriever(device=config.get("embedding", {}).get("device", "cuda"))
        _ = retriever.num_chunks
        if len(questions) > 0:
            _ = retriever.retrieve(
                questions[0]["question"], k=1,
                include_options=questions[0].get("options")
            )
        print(f"[Warmup] Retriever ready: {retriever.num_chunks} chunks, {len(retriever.sources)} sources")

    print(f"\n🏥 Running {variant.upper()} on {split} ({len(questions)} questions)")
    print(f"   Memory: {'ON' if with_memory else 'OFF'} | Verifier: {'ON' if with_verifier else 'OFF'}")
    print(f"   Model: {config['model']['provider']}/{config['model']['name']}\n")

    results = []
    all_traces = []
    for i, q in enumerate(questions):
        state = initial_state(q, config)
        # Dynamic recursion limit: 5 (startup) + max_rounds * 3 (per round) + 10 (buffer)
        max_rounds = config.get("pipeline", {}).get("max_rounds", 8)
        recursion_limit = max_rounds * 4 + 15
        try:
            final = graph.invoke(state, {"recursion_limit": recursion_limit})
        except Exception as e:
            print(f"  ❌ ERROR on {q['question_id']}: {e}")
            final = None
            results.append({
                "question_id": q["question_id"],
                "meta_info": q.get("meta_info", ""),
                "predicted_answer": None,
                "ground_truth": q["answer_idx"],
                "correct": False,
                "variant": variant,
                "error": str(e),
                "latency": 0,
                "rounds_used": 0,
            })
            all_traces.append({
                "question_id": q["question_id"],
                "variant": variant,
                "predicted_answer": None,
                "ground_truth": q["answer_idx"],
                "correct": False,
                "error": str(e),
                "node_traces": [],
            })
            continue

        predicted = _extract_answer(final)

        ground_truth = q["answer_idx"]
        correct = predicted == ground_truth if predicted else False

        # Record to LTM if correct
        if with_memory and correct:
            record_question(
                question_id=q["question_id"],
                question_stem=q["question"],
                ground_truth=ground_truth,
                rag_context=final.get("rag_context", ""),
                explanation=final.get("candidate_explanation", ""),
                predicted_answer=predicted,
            )

        result = {
            "question_id": q["question_id"],
            "meta_info": q.get("meta_info", ""),
            "predicted_answer": predicted,
            "ground_truth": ground_truth,
            "correct": correct,
            "variant": variant,
            "candidate": predicted or "",
            "verdict": final.get("verifier_verdict", ""),
            "verifier_reason": final.get("verifier_reason", ""),
            "verifier_confidence": final.get("verifier_confidence", 0.0),
            "rounds_used": final.get("rounds_used", 0),
            "degraded": final.get("degraded", False),
            "latency": round(final.get("total_latency", 0), 3),
            "input_tokens": final.get("total_input_tokens", 0),
            "output_tokens": final.get("total_output_tokens", 0),
            "rag_source": final.get("rag_source", ""),
            "ltm_found": len(final.get("raw_ltm_entries", [])),
        }
        results.append(result)

        # Save trace for re-evaluation
        trace_entry = {
            "question_id": q["question_id"],
            "variant": variant,
            "predicted_answer": predicted,
            "ground_truth": ground_truth,
            "correct": correct,
            "node_traces": final.get("node_traces", []),
        }
        all_traces.append(trace_entry)

        # Progress
        if (i + 1) % 5 == 0 or i == 0:
            running_acc = sum(1 for r in results if r["correct"])
            print(f"  [{i+1}/{len(questions)}] Acc: {running_acc/len(results):.3f} | "
                  f"{' ✓' if correct else ' ✗'} | Pred: {predicted} Truth: {ground_truth} | "
                  f"Rounds: {final.get('rounds_used', 0)}")

    # Save predictions
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"predictions_{variant}_{split}_{timestamp}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Predictions saved: {out_path.name}")

    # Save detailed traces (per-node I/O for re-evaluation)
    traces_path = OUTPUT_DIR / f"traces_{variant}_{split}_{timestamp}.jsonl"
    with open(traces_path, "w") as f:
        for t in all_traces:
            f.write(json.dumps(t, default=str) + "\n")
    print(f"  Traces saved: {traces_path.name}")

    # Print summary
    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0
    avg_latency = sum(r.get("latency", 0) for r in results) / len(results) if results else 0
    avg_rounds = sum(r.get("rounds_used", 0) for r in results) / len(results) if results else 0

    print(f"\n{'='*60}")
    print(f"  {variant.upper()} — {split} ({len(results)} questions)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy:.4f} ({int(accuracy * len(results))}/{len(results)})")
    print(f"  Avg rounds: {avg_rounds:.1f}")
    print(f"  Avg latency: {avg_latency:.2f}s")
    if with_verifier:
        approves = sum(1 for r in results if r.get("verdict") == "APPROVE")
        retries = sum(1 for r in results if r.get("verdict") == "RETRY")
        print(f"  Verifier: {approves} APPROVE / {retries} RETRY")
    print(f"  LTM size: {len(_ltm_store_summary())}" if with_memory else "")
    print(f"  Saved: {out_path.name}")
    print(f"{'='*60}\n")

    return results


def _extract_answer(final: dict) -> Optional[str]:
    """Extract final answer from graph output."""
    candidate = final.get("candidate_letter", "") or final.get("candidate_answer", "")
    if candidate and candidate in "ABCD":
        return candidate
    # V4 fallback (no verifier)
    candidate = final.get("candidate_answer", "")
    if candidate and candidate in "ABCD":
        return candidate
    return None


def _ltm_store_summary() -> list[dict]:
    """Get current LTM store size (for display)."""
    from medqa_usmle.agents.memory import _LTM_STORE
    return _LTM_STORE


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V3: Full Multi-Agent System")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument("--no-clear-ltm", action="store_true",
                        help="Keep existing LTM from previous runs")
    args = parser.parse_args()
    run_v3(
        split=args.split, limit=args.limit, max_samples=args.max_samples,
        with_memory=not args.no_memory, with_verifier=not args.no_verifier,
        clear_ltm_before=not args.no_clear_ltm,
    )
