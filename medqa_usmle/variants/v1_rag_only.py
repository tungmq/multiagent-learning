"""
V1: RAG-only Baseline

Single LLM call with RAG context injected (no multi-agent).
Isolates the contribution of RAG knowledge augmentation.
"""

import datetime
import json
import time
from pathlib import Path
from typing import Optional

import requests

from medqa_usmle.configs import load_config
from medqa_usmle.data.loader import load_questions, format_question, format_options_text
from medqa_usmle.llm import create_provider, BaseProvider

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a medical student taking the USMLE Step exam. "
    "Use the provided medical knowledge to help answer the question. "
    "Respond with ONLY a JSON object on one line: "
    '{"answer": "B", "explanation": "Brief reasoning"}. '
    "The answer MUST be one of: A, B, C, or D."
)


def build_prompt(question: dict, rag_context: str = "") -> str:
    """Build prompt with optional RAG context."""
    parts = [f"Question: {question['question']}"]
    parts.append(f"\nOptions:\n{format_options_text(question)}")
    if rag_context:
        parts.append(f"\nMedical Knowledge:\n{rag_context}")
    return "\n".join(parts)


def run_single_question(provider: BaseProvider, question: dict, rag_context: str = "", **kwargs) -> dict:
    prompt = build_prompt(question, rag_context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Retry up to 3 times on transient API errors
    last_error = None
    for attempt in range(3):
        try:
            resp = provider.invoke(messages, **kwargs)
            break
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < 2:
                wait = 10 * (attempt + 1)
                print(f"  ⚠ Retry {attempt+1}/3 after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise last_error
    content = resp.content.strip()

    # Extract answer from JSON
    predicted = None
    explanation = ""
    if content:
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                predicted = parsed.get("answer", "")
                explanation = parsed.get("explanation", "") or parsed.get("reason", "")
        except (json.JSONDecodeError, ValueError):
            for letter in ["A", "B", "C", "D"]:
                if f'"{letter}"' in content or letter in content.upper():
                    predicted = letter
                    break

    ground_truth = question["answer_idx"]
    correct = predicted == ground_truth if predicted else False

    return {
        "question_id": question["question_id"],
        "meta_info": question.get("meta_info", ""),
        "predicted_answer": predicted,
        "ground_truth": ground_truth,
        "correct": correct,
        "latency": round(resp.latency, 3),
        "input_tokens": resp.usage.get("input_tokens", 0),
        "output_tokens": resp.usage.get("output_tokens", 0),
        "variant": "v1",
        # Trace support (Web UI): what was sent to the model and what came back
        "messages_sent": messages,
        "raw_response": resp.content,
        "rag_context": rag_context,
    }


def run_v1(
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
        # Simple RAG: inject a general medical knowledge prompt
        # Will be replaced with Qdrant retrieval when available
        rag_context = ("Consider the clinical presentation, differential diagnoses, "
                      "and standard management guidelines for this condition.")

        result = run_single_question(provider, q, rag_context=rag_context)
        results.append(result)

        if (i + 1) % 5 == 0 or i == 0:
            acc = sum(1 for r in results if r["correct"]) / len(results)
            print(f"  [{i+1}/{len(questions)}] Acc: {acc:.3f}")

        # Incremental save every 25 questions to avoid data loss on crash
        if (i + 1) % 25 == 0:
            inc_path = OUTPUT_DIR / f"predictions_v1_{split}_inc.jsonl"
            with open(inc_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"predictions_v1_{split}_{timestamp}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0
    print(f"\n=== V1 Results ({split}, n={len(results)}) ===")
    print(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(results))}/{len(results)})")
    print(f"Saved to: {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V1: RAG-only")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    run_v1(split=args.split, max_samples=args.max_samples, limit=args.limit)
