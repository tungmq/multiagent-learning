"""
V4 Parallel Runner — V3 minus Verifier (ablation).

V4 = Memory → RAG → Reasoning (no verifier gate).
Purpose: measure accuracy without verifier to isolate its impact.
Using new config: num_predict=8192 (same as V3 new config).
"""
import datetime
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Optional

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load API key before anything else
env_path = Path.home() / ".hermes" / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENCODE_GO_API_KEY"):
                key = line.split("=", 1)[1].strip().strip("'\"").strip('"')
                os.environ["OPENCODE_GO_API_KEY"] = key
                break

from medqa_usmle.configs import load_config
from medqa_usmle.data.loader import load_questions
from medqa_usmle.agents.state import initial_state
from medqa_usmle.variants.v3_full_system import (
    build_variant_graph, _extract_answer, _ltm_store_summary, clear_ltm,
)
from medqa_usmle.agents.memory import record_question

OUTPUT_DIR = Path(PROJECT_ROOT) / "medqa_usmle" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_WORKERS = 6

# Thread-safe LTM lock
_ltm_lock = threading.Lock()


def process_question(q: dict, graph, config: dict, variant: str) -> dict:
    """Process a single question through the V4 pipeline (no verifier)."""
    state = initial_state(q, config)
    max_rounds = config.get("pipeline", {}).get("max_rounds", 5)
    recursion_limit = max_rounds * 4 + 15

    try:
        final = graph.invoke(state, {"recursion_limit": recursion_limit})
    except Exception as e:
        return {
            "question_id": q["question_id"],
            "meta_info": q.get("meta_info", ""),
            "predicted_answer": None,
            "ground_truth": q["answer_idx"],
            "correct": False,
            "variant": variant,
            "error": str(e),
            "latency": 0,
            "rounds_used": 0,
        }

    predicted = _extract_answer(final)
    ground_truth = q["answer_idx"]
    correct = predicted == ground_truth if predicted else False

    # Record to LTM (thread-safe)
    if correct:
        with _ltm_lock:
            record_question(
                question_id=q["question_id"],
                question_stem=q["question"],
                ground_truth=ground_truth,
                rag_context=final.get("rag_context", ""),
                explanation=final.get("candidate_explanation", ""),
                predicted_answer=predicted,
            )

    return {
        "question_id": q["question_id"],
        "meta_info": q.get("meta_info", ""),
        "predicted_answer": predicted,
        "ground_truth": ground_truth,
        "correct": correct,
        "variant": variant,
        "candidate": predicted or "",
        "rounds_used": final.get("rounds_used", 0),
        "degraded": final.get("degraded", False),
        "latency": round(final.get("total_latency", 0), 3),
        "input_tokens": final.get("total_input_tokens", 0),
        "output_tokens": final.get("total_output_tokens", 0),
        "rag_source": final.get("rag_source", ""),
        "ltm_found": len(final.get("raw_ltm_entries", [])),
    }


QUESTION_TIMEOUT = 120  # 2 min per question before skipping


def run_v4_parallel(
    split: str = "test",
    limit: Optional[int] = None,
    max_samples: Optional[int] = None,
    workers: int = NUM_WORKERS,
) -> list[dict]:
    """Run V4 (no verifier) on MedQA-USMLE with parallel question processing."""
    config = load_config()
    variant = "v4"

    # Clear LTM
    clear_ltm()

    # Build graph once — V4 = with_memory=True, with_verifier=False
    graph = build_variant_graph(config, with_memory=True, with_verifier=False)

    # Load questions
    questions = load_questions(split, max_samples=max_samples)
    if limit:
        questions = questions[:limit]

    # Warm up retriever
    from medqa_usmle.rag.retriever import MedicalRetriever
    print(f"[Warmup] Initializing MedicalRetriever...", flush=True)
    retriever = MedicalRetriever(device=config.get("embedding", {}).get("device", "cuda"))
    _ = retriever.num_chunks
    if len(questions) > 0:
        _ = retriever.retrieve(questions[0]["question"], k=1,
                               include_options=questions[0].get("options"))
    print(f"[Warmup] Retriever ready: {retriever.num_chunks} chunks, {len(retriever.sources)} sources", flush=True)

    print(f"\n🏥 Running V4 (no verifier) — {split} ({len(questions)} questions, {workers} workers, max_rounds={config['pipeline']['max_rounds']})", flush=True)
    print(f"   Model: {config['model']['provider']}/{config['model']['name']}", flush=True)
    print(f"   Per-question timeout: {QUESTION_TIMEOUT}s", flush=True)
    start_time = datetime.datetime.now()

    results = []
    completed = 0
    results_lock = threading.Lock()

    def _process_and_report(q):
        result = process_question(q, graph, config, variant)
        return result, q

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_process_and_report, q): q for q in questions}

        for future in as_completed(future_map):
            q = future_map[future]
            try:
                result, q_obj = future.result(timeout=QUESTION_TIMEOUT)
            except TimeoutError:
                result = {
                    "question_id": q["question_id"],
                    "meta_info": q.get("meta_info", ""),
                    "predicted_answer": None,
                    "ground_truth": q["answer_idx"],
                    "correct": False,
                    "variant": variant,
                    "error": "timeout",
                    "latency": 0,
                    "rounds_used": 0,
                }
            except Exception as e:
                result = {
                    "question_id": q["question_id"],
                    "meta_info": q.get("meta_info", ""),
                    "predicted_answer": None,
                    "ground_truth": q["answer_idx"],
                    "correct": False,
                    "variant": variant,
                    "error": str(e)[:80],
                    "latency": 0,
                    "rounds_used": 0,
                }

            with results_lock:
                results.append(result)
                completed += 1
                running_acc = sum(1 for r in results if r["correct"])
                pct = completed / len(questions) * 100

                mark = " ✓" if result["correct"] else " ✗"
                error = result.get("error", "")
                if error:
                    mark = " ⏱️" if "timeout" in error else " ❌"

                print(f"  [{completed}/{len(questions)}] {pct:5.1f}% | Acc: {running_acc}/{completed} = {running_acc/completed:.3f} |"
                      f"{mark} | {q['question_id']} | Pred:{result['predicted_answer']} Truth:{result['ground_truth']} | R:{result.get('rounds_used',0)}{' ERR:'+error[:40] if error else ''}", flush=True)

    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0
    avg_latency = sum(r.get("latency", 0) for r in results) / len(results) if results else 0
    avg_rounds = sum(r.get("rounds_used", 0) for r in results) / len(results) if results else 0

    print(f"\n{'='*60}", flush=True)
    print(f"  V4 (no verifier) — {split} ({len(results)} questions, {workers} workers)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Accuracy:  {accuracy:.4f} ({int(accuracy * len(results))}/{len(results)})", flush=True)
    print(f"  Avg rounds: {avg_rounds:.1f}", flush=True)
    print(f"  Avg latency: {avg_latency:.2f}s", flush=True)
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    errors = sum(1 for r in results if r.get("error"))
    print(f"  Errors: {errors}", flush=True)
    print(f"  LTM size: {len(_ltm_store_summary())}", flush=True)

    # Save predictions
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"predictions_{variant}_{split}_{timestamp}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Saved: {out_path.name}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V4 Parallel Runner (no verifier)")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    run_v4_parallel(
        split=args.split,
        limit=args.limit,
        max_samples=args.max_samples,
        workers=args.workers,
    )
