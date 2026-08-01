"""
V3 Rerun v2: increase recursion_limit to 40 and timeout to 300s.

Re-run the questions that timed out in v1 rerun.
"""
import datetime
import json
import signal
from pathlib import Path

from medqa_usmle.configs import load_config
from medqa_usmle.data.loader import load_questions
from medqa_usmle.agents.state import initial_state
from medqa_usmle.agents.memory import clear_ltm, record_question

from v3_full_system import build_variant_graph, _extract_answer

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECURSION_LIMIT = 40
TIMEOUT_SECS = 300


class TimeoutError_(Exception):
    pass


def _timeout_handler(_signum, _frame):
    raise TimeoutError_(f"Graph invoke timed out after {TIMEOUT_SECS}s")


def get_failed_ids(predictions_path: str, error_types: set[str] = None) -> list[str]:
    """Extract question IDs that had specific error types."""
    errored = []
    with open(predictions_path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("correct"):
                err = d.get("error", "")
                if error_types is None:
                    errored.append(d["question_id"])
                elif any(t in err for t in error_types):
                    errored.append(d["question_id"])
    return errored


def load_specific_questions(split: str, question_ids: set[str]) -> list[dict]:
    all_qs = load_questions(split)
    return [q for q in all_qs if q["question_id"] in question_ids]


def run_rerun_v2(
    predictions_path: str,
    split: str = "test",
    with_memory: bool = True,
    with_verifier: bool = True,
):
    config = load_config()

    # Get failed IDs from the rerun v1: timeout + other wrong
    failed_ids = get_failed_ids(
        predictions_path,
        error_types={"timed out"},
    )
    print(f"Found {len(failed_ids)} failed question IDs")

    if len(failed_ids) == 0:
        print("No questions to rerun!")
        return

    questions = load_specific_questions(split, set(failed_ids))
    print(f"Loaded {len(questions)} questions for rerun v2")

    clear_ltm()

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
            include_options=questions[0].get("options"),
        )
    print(f"[Warmup] Retriever ready: {retriever.num_chunks} chunks, {len(retriever.sources)} sources")

    print(f"\n🏥 Running V3 RERUN v2 on {split} ({len(questions)} questions)")
    print(f"   Memory: ON | Verifier: ON | Recursion: {RECURSION_LIMIT} | Timeout: {TIMEOUT_SECS}s")
    print(f"   Model: {config['model']['provider']}/{config['model']['name']}\n")

    signal.signal(signal.SIGALRM, _timeout_handler)

    results = []
    all_traces = []
    for i, q in enumerate(questions):
        state = initial_state(q, config)
        try:
            signal.alarm(TIMEOUT_SECS)
            try:
                final = graph.invoke(state, {"recursion_limit": RECURSION_LIMIT})
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
                "rerun_v2": True,
            })
            all_traces.append({
                "question_id": q["question_id"],
                "variant": variant,
                "predicted_answer": None,
                "ground_truth": q["answer_idx"],
                "correct": False,
                "error": str(e),
                "node_traces": [],
                "rerun_v2": True,
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
            "rerun_v2": True,
        }
        results.append(result)

        trace_entry = {
            "question_id": q["question_id"],
            "variant": variant,
            "predicted_answer": predicted,
            "ground_truth": ground_truth,
            "correct": correct,
            "node_traces": final.get("node_traces", []),
            "rerun_v2": True,
        }
        all_traces.append(trace_entry)

        if (i + 1) % 5 == 0 or i == 0:
            running_acc = sum(1 for r in results if r["correct"])
            print(f"  [{i+1}/{len(questions)}] Acc: {running_acc/len(results):.3f} | "
                  f"{' ✓' if correct else ' ✗'} | Pred: {predicted} Truth: {ground_truth} | "
                  f"Rounds: {final.get('rounds_used', 0)} | "
                  f"Lat: {final.get('total_latency', 0):.1f}s")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"predictions_v3_rerun_v2_{split}_{timestamp}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Predictions saved: {out_path.name}")

    traces_path = OUTPUT_DIR / f"traces_v3_rerun_v2_{split}_{timestamp}.jsonl"
    with open(traces_path, "w") as f:
        for t in all_traces:
            f.write(json.dumps(t, default=str) + "\n")
    print(f"  Traces saved: {traces_path.name}")

    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0
    avg_latency = sum(r.get("latency", 0) for r in results) / len(results) if results else 0
    avg_rounds = sum(r.get("rounds_used", 0) for r in results) / len(results) if results else 0
    approves = sum(1 for r in results if r.get("verdict") == "APPROVE")
    retries = sum(1 for r in results if r.get("verdict") == "RETRY")
    timeout_cnt = sum(1 for r in results if "timed out" in r.get("error", ""))

    print(f"\n{'='*60}")
    print(f"  V3 RERUN v2 — {split} ({len(results)} questions)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy:.4f} ({int(accuracy * len(results))}/{len(results)})")
    print(f"  Timeout:   {timeout_cnt}")
    print(f"  Avg rounds: {avg_rounds:.1f}")
    print(f"  Avg latency: {avg_latency:.2f}s")
    print(f"  Verifier: {approves} APPROVE / {retries} RETRY")
    print(f"  Saved: {out_path.name}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V3 Rerun v2: higher recursion + longer timeout")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSONL from rerun v1")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args()

    run_rerun_v2(
        predictions_path=args.predictions,
        split=args.split,
    )
