"""
V0: Direct LLM Baseline

Single LLM call, no tools, no RAG, no memory, no multi-agent.
Uses the provider abstraction layer (Ollama / OpenAI-compatible).

Purpose: Establish performance floor for comparison with V1-V4.
"""

import datetime
import json
import time
from pathlib import Path
from typing import Optional

from medqa_usmle.configs import load_config
from medqa_usmle.data.loader import load_questions, format_question
from medqa_usmle.llm import create_provider, BaseProvider

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a medical student taking the USMLE Step exam. "
    "Answer the following multiple choice question with the single best letter (A, B, C, or D). "
    "Respond with ONLY the letter. No explanation."
)


def run_single_question(provider: BaseProvider, question: dict, **kwargs) -> dict:
    formatted = format_question(question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted},
    ]

    resp = provider.invoke(messages, **kwargs)
    raw = resp.content.strip().upper()

    # Extract answer letter
    predicted = None
    for letter in ["A", "B", "C", "D"]:
        if letter in raw:
            predicted = letter
            break
    if predicted is None and raw in ["A", "B", "C", "D"]:
        predicted = raw

    ground_truth = question["answer_idx"]
    correct = predicted == ground_truth if predicted else False

    return {
        "question_id": question["question_id"],
        "meta_info": question.get("meta_info", ""),
        "predicted_answer": predicted,
        "ground_truth": ground_truth,
        "correct": correct,
        "latency": round(resp.latency, 3),
        "raw_output": raw,
        "variant": "v0",
        "input_tokens": resp.usage.get("input_tokens", 0),
        "output_tokens": resp.usage.get("output_tokens", 0),
        # Trace support (Web UI): what was sent to the model and what came back
        "messages_sent": messages,
        "raw_response": resp.content,
    }


def run_v0(
    split: str = "dev",
    max_samples: Optional[int] = None,
    limit: Optional[int] = None,
    config_override: Optional[dict] = None,
) -> list[dict]:
    config = config_override or load_config()
    provider = create_provider(config)
    questions = load_questions(split, max_samples=max_samples)
    if limit:
        questions = questions[:limit]

    results = []
    for i, q in enumerate(questions):
        result = run_single_question(provider, q)
        results.append(result)
        if (i + 1) % 20 == 0:
            acc = sum(1 for r in results if r["correct"]) / len(results)
            print(f"  [{i+1}/{len(questions)}] Running acc: {acc:.3f}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"predictions_v0_{split}_{timestamp}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0
    total_latency = sum(r["latency"] for r in results)
    total_input = sum(r.get("input_tokens", 0) for r in results)
    total_output = sum(r.get("output_tokens", 0) for r in results)

    print(f"\n=== V0 Results ({split}, n={len(results)}) ===")
    print(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(results))}/{len(results)})")
    print(f"Avg latency: {total_latency/len(results):.2f}s" if results else "N/A")
    print(f"Total tokens: {total_input} in / {total_output} out")
    print(f"Saved to: {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V0: Direct LLM Baseline")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    run_v0(split=args.split, max_samples=args.max_samples, limit=args.limit)
