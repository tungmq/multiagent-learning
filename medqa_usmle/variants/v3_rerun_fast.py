"""
Fast rerun for errored questions: skip verifier to avoid slow retry loops.
Uses V4-style pipeline (memory→rag→reasoning→END) for the errored questions.
"""
import datetime
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medqa_usmle.configs import load_config
from medqa_usmle.data.loader import load_questions
from medqa_usmle.agents.state import initial_state
from medqa_usmle.agents.memory import clear_ltm, record_question
from medqa_usmle.variants.v3_full_system import build_variant_graph, _extract_answer
from medqa_usmle.rag.retriever import MedicalRetriever

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECURSION_LIMIT = 50
TIMEOUT_SECS = 180


def get_errored_ids(predictions_path: str) -> list[str]:
    """Extract question IDs that had errors (recursion limit, API 500, etc)."""
    errored = []
    with open(predictions_path) as f:
        for line in f:
            d = json.loads(line)
            if "error" in d:
                errored.append(d["question_id"])
    return errored


def run_fast_rerun(
    predictions_path: str,
    split: str = "test",
    with_memory: bool = True,
    no_verifier: bool = True,
):
    config = load_config()
    # Set max_rounds high enough for initial pipeline
    config.setdefault("pipeline", {})["max_rounds"] = 8

    # Find errored IDs
    errored_ids = get_errored_ids(predictions_path)
    print(f"Found {len(errored_ids)} errored question IDs", flush=True)

    # Load only those questions
    all_qs = load_questions(split)
    questions = [q for q in all_qs if q["question_id"] in set(errored_ids)]
    print(f"Loaded {len(questions)} questions for rerun", flush=True)

    if len(questions) == 0:
        print("No questions to rerun!", flush=True)
        return

    # Start fresh LTM
    clear_ltm()

    # Build graph (no verifier for speed)
    graph = build_variant_graph(config, with_memory=with_memory, with_verifier=not no_verifier)
    variant = "v3_rerun"

    # Pre-warm retriever singleton
    print("[Warmup] Pre-warming retriever...", flush=True)
    t0 = time.time()
    retriever = MedicalRetriever(device=config.get("embedding", {}).get("device", "cuda"))
    _ = retriever.num_chunks
    if len(questions) > 0:
        _ = retriever.retrieve(
            questions[0]["question"], k=1,
            include_options=questions[0].get("options")
        )
    print(f"[Warmup] Retriever ready: {retriever.num_chunks} chunks ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n🏥 Running V3 FAST RERUN on {split} ({len(questions)} errored questions)", flush=True)
    print(f"   Memory: {'ON' if with_memory else 'OFF'} | Verifier: OFF (fast mode)", flush=True)
    print(f"   Model: {config['model']['provider']}/{config['model']['name']}\n", flush=True)

    class TimeoutError_(Exception):
        pass

    def _timeout_handler(_signum, _frame):
        raise TimeoutError_("Graph invoke timed out")

    signal.signal(signal.SIGALRM, _timeout_handler)

    results = []
    all_traces = []
    for i, q in enumerate(questions):
        state = initial_state(q, config)
        t_start = time.time()
        try:
            signal.alarm(TIMEOUT_SECS)
            try:
                final = graph.invoke(state, {"recursion_limit": RECURSION_LIMIT})
            finally:
                signal.alarm(0)
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(questions)}] ❌ ERROR on {q['question_id']}: {e} ({elapsed:.0f}s)", flush=True)
            results.append({
                "question_id": q["question_id"],
                "meta_info": q.get("meta_info", ""),
                "predicted_answer": None,
                "ground_truth": q["answer_idx"],
                "correct": False,
                "variant": variant,
                "error": str(e),
                "latency": elapsed,
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

        elapsed = time.time() - t_start
        result = {
            "question_id": q["question_id"],
            "meta_info": q.get("meta_info", ""),
            "predicted_answer": predicted,
            "ground_truth": ground_truth,
            "correct": correct,
            "variant": variant,
            "rounds_used": final.get("rounds_used", 0),
            "degraded": final.get("degraded", False),
            "latency": round(elapsed, 3),
            "input_tokens": final.get("total_input_tokens", 0),
            "output_tokens": final.get("total_output_tokens", 0),
            "rag_source": final.get("rag_source", ""),
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

        running_acc = sum(1 for r in results if r["correct"])
        print(f"  [{i+1}/{len(questions)}] {q['question_id']} | {'✓' if correct else '✗'} | "
              f"Pred:{predicted} Truth:{ground_truth} | R:{final.get('rounds_used', 0)} | "
              f"{elapsed:.0f}s | Acc:{running_acc}/{i+1}={running_acc/(i+1):.3f}", flush=True)

        # Save incremental after each question
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"predictions_v3_rerun_{split}_{timestamp}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        traces_path = OUTPUT_DIR / f"traces_v3_rerun_{split}_{timestamp}.jsonl"
        with open(traces_path, "w") as f:
            for t in all_traces:
                f.write(json.dumps(t, default=str) + "\n")

    # Final summary
    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0
    avg_latency = sum(r.get("latency", 0) for r in results) / len(results) if results else 0
    print(f"\n{'='*60}", flush=True)
    print(f"  V3 FAST RERUN — {split} ({len(results)} questions)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Accuracy:  {accuracy:.4f} ({int(accuracy * len(results))}/{len(results)})", flush=True)
    print(f"  Avg latency: {avg_latency:.2f}s", flush=True)
    print(f"{'='*60}\n", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V3 Fast Rerun (no verifier)")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSONL from previous run")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--no-memory", action="store_true")
    args = parser.parse_args()

    run_fast_rerun(
        predictions_path=args.predictions,
        split=args.split,
        with_memory=not args.no_memory,
        no_verifier=True,
    )
