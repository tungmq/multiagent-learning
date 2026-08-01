"""
V3 Rerun: re-run only the errored questions (recursion limit + API 500).

Loads errored question IDs from a previous predictions file,
filters the test split to just those questions, and runs V3 on them.
Saves a separate predictions file.
"""
import datetime
import json
from pathlib import Path

from medqa_usmle.configs import load_config
from medqa_usmle.data.loader import load_questions
from medqa_usmle.agents.state import initial_state
from medqa_usmle.agents.memory import clear_ltm, record_question

from v3_full_system import build_variant_graph, _extract_answer

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_errored_ids(predictions_path: str) -> list[str]:
    """Extract question IDs that had errors (recursion limit, API 500, etc)."""
    errored = []
    with open(predictions_path) as f:
        for line in f:
            d = json.loads(line)
            if "error" in d:
                errored.append(d["question_id"])
            elif not d.get("correct"):
                # Also include wrong answers that have empty verdict (error-like)
                if not d.get("verdict"):
                    errored.append(d["question_id"])
    return errored


def load_specific_questions(split: str, question_ids: set[str]) -> list[dict]:
    """Load only the questions matching given IDs."""
    all_qs = load_questions(split)
    filtered = [q for q in all_qs if q["question_id"] in question_ids]
    return filtered


def run_rerun(
    predictions_path: str,
    split: str = "test",
    with_memory: bool = True,
    with_verifier: bool = True,
):
    config = load_config()
    # Override for rerun: tracking retries via coordinator, so max_rounds is now effective.
    # Keep at 5 for reasonable throughput (initial pipeline + 1 retry).

    # Find errored IDs
    errored_ids = get_errored_ids(predictions_path)
    print(f"Found {len(errored_ids)} errored question IDs")

    # Load only those questions
    questions = load_specific_questions(split, set(errored_ids))
    print(f"Loaded {len(questions)} questions for rerun")

    if len(questions) == 0:
        print("No questions to rerun!")
        return

    # Start fresh LTM (previous run's in-memory LTM is gone anyway)
    clear_ltm()

    # Build graph
    graph = build_variant_graph(config, with_memory=with_memory, with_verifier=with_verifier)
    variant = "v3"

    # Warm up retriever
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

    print(f"\n🏥 Running V3 RERUN on {split} ({len(questions)} errored questions)")
    print(f"   Memory: {'ON' if with_memory else 'OFF'} | Verifier: {'ON' if with_verifier else 'OFF'}")
    print(f"   Model: {config['model']['provider']}/{config['model']['name']}\n")

    results = []
    all_traces = []
    import signal

    class TimeoutError_(Exception):
        pass

    def _timeout_handler(_signum, _frame):
        raise TimeoutError_("Graph invoke timed out")

    for i, q in enumerate(questions):
        state = initial_state(q, config)
        try:
            # Set alarm for 180s timeout
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(180)
            try:
                final = graph.invoke(state, {"recursion_limit": 50})
            finally:
                signal.alarm(0)
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
            "rerun": True,
        }
        results.append(result)

        trace_entry = {
            "question_id": q["question_id"],
            "variant": variant,
            "predicted_answer": predicted,
            "ground_truth": ground_truth,
            "correct": correct,
            "node_traces": final.get("node_traces", []),
            "rerun": True,
        }
        all_traces.append(trace_entry)

        if (i + 1) % 5 == 0 or i == 0:
            running_acc = sum(1 for r in results if r["correct"])
            print(f"  [{i+1}/{len(questions)}] Acc: {running_acc/len(results):.3f} | "
                  f"{' ✓' if correct else ' ✗'} | Pred: {predicted} Truth: {ground_truth} | "
                  f"Rounds: {final.get('rounds_used', 0)}")

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"predictions_v3_rerun_{split}_{timestamp}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Predictions saved: {out_path.name}")

    traces_path = OUTPUT_DIR / f"traces_v3_rerun_{split}_{timestamp}.jsonl"
    with open(traces_path, "w") as f:
        for t in all_traces:
            f.write(json.dumps(t, default=str) + "\n")
    print(f"  Traces saved: {traces_path.name}")

    # Summary
    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0
    avg_latency = sum(r.get("latency", 0) for r in results) / len(results) if results else 0
    avg_rounds = sum(r.get("rounds_used", 0) for r in results) / len(results) if results else 0
    approves = sum(1 for r in results if r.get("verdict") == "APPROVE")
    retries = sum(1 for r in results if r.get("verdict") == "RETRY")

    print(f"\n{'='*60}")
    print(f"  V3 RERUN — {split} ({len(results)} questions)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy:.4f} ({int(accuracy * len(results))}/{len(results)})")
    print(f"  Avg rounds: {avg_rounds:.1f}")
    print(f"  Avg latency: {avg_latency:.2f}s")
    print(f"  Verifier: {approves} APPROVE / {retries} RETRY")
    print(f"  Saved: {out_path.name}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V3 Rerun errored questions")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSONL from previous run")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-verifier", action="store_true")
    args = parser.parse_args()

    run_rerun(
        predictions_path=args.predictions,
        split=args.split,
        with_memory=not args.no_memory,
        with_verifier=not args.no_verifier,
    )
