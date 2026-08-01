"""
MedQA-USMLE Data Loader.

Loads questions from the MedQA-USMLE dataset (US/4_options jsonl files).
Returns structured question dicts with fields: question, answer, options, answer_idx, meta_info.
"""

import json
import random
from pathlib import Path
from typing import Optional

# Dataset paths relative to repo root
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "MedQA-USMLE" / "questions" / "US" / "4_options"

DEV_FILE = DATASET_DIR / "phrases_no_exclude_dev.jsonl"
TEST_FILE = DATASET_DIR / "phrases_no_exclude_test.jsonl"


def load_questions(
    split: str = "dev",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Load MedQA-USMLE questions.

    Args:
        split: "dev" (1272 questions) or "test" (1273 questions)
        max_samples: If set, randomly sample this many questions (for quick dev)
        seed: Random seed for sampling

    Returns:
        List of dicts with keys:
            question_id: str (e.g., "medqa_dev_0000")
            question: str (stem)
            options: dict[str, str] (e.g., {"A": "...", "B": "..."})
            answer: str (ground truth text)
            answer_idx: str (A/B/C/D)
            meta_info: str (step1 or step2&3)
    """
    path = DEV_FILE if split == "dev" else TEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            questions.append({
                "question_id": f"medqa_{split}_{i:04d}",
                "question": item["question"],
                "options": item["options"],  # dict[str, str]
                "answer": item["answer"],
                "answer_idx": item["answer_idx"],  # A/B/C/D
                "meta_info": item.get("meta_info", "unknown"),
            })

    if max_samples is not None and max_samples < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, max_samples)

    return questions


def format_question(q: dict) -> str:
    """Format a question for LLM prompt input."""
    lines = [f"Question: {q['question']}"]
    for letter in ["A", "B", "C", "D"]:
        if letter in q["options"]:
            lines.append(f"{letter}. {q['options'][letter]}")
    return "\n".join(lines)


def format_options_text(q: dict) -> str:
    """Format just the options portion."""
    lines = []
    for letter in ["A", "B", "C", "D"]:
        if letter in q["options"]:
            lines.append(f"{letter}. {q['options'][letter]}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    dev = load_questions("dev", max_samples=5)
    print(f"Loaded {len(dev)} dev questions (sample)")
    for q in dev:
        print(f"\n--- {q['question_id']} ({q['meta_info']}) ---")
        print(f"Q: {q['question'][:80]}...")
        print(f"Options: {list(q['options'].keys())}")
        print(f"Answer: {q['answer_idx']} — {q['answer'][:60]}...")
        print(f"Formatted:\\n{format_question(q)[:200]}...")

    test = load_questions("test", max_samples=3)
    print(f"\nLoaded {len(test)} test questions (sample)")
