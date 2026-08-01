"""
Evaluation Grader for MedQA-USMLE.

Metrics:
- Accuracy
- Invalid Response Rate
- Accuracy Gain (V3 vs V0)
- Win/Loss/Tie per question
- Cost & Latency
- Error breakdown
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional


def load_predictions(path: str) -> list[dict]:
    """Load predictions from JSONL file."""
    results = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_accuracy(results: list[dict]) -> float:
    """Compute accuracy (correct / total)."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("correct")) / len(results)


def compute_invalid_rate(results: list[dict]) -> float:
    """Rate of invalid responses (not A/B/C/D)."""
    if not results:
        return 0.0
    invalid = sum(1 for r in results if r.get("predicted_answer") not in ("A", "B", "C", "D"))
    return invalid / len(results)


def compute_win_loss_tie(v3_results: list[dict], v0_results: list[dict]) -> dict:
    """Per-question V3 vs V0 comparison."""
    v3_map = {r["question_id"]: r for r in v3_results}
    v0_map = {r["question_id"]: r for r in v0_results}
    
    wins = losses = ties = 0
    for qid in set(v3_map) & set(v0_map):
        v3_correct = v3_map[qid].get("correct", False)
        v0_correct = v0_map[qid].get("correct", False)
        if v3_correct and not v0_correct:
            wins += 1
        elif not v3_correct and v0_correct:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def compute_stratified(results: list[dict]) -> dict:
    """Accuracy stratified by meta_info."""
    by_meta = {}
    for r in results:
        meta = r.get("meta_info", "unknown")
        if meta not in by_meta:
            by_meta[meta] = {"correct": 0, "total": 0}
        by_meta[meta]["total"] += 1
        if r.get("correct"):
            by_meta[meta]["correct"] += 1
    return {
        meta: {
            "accuracy": vals["correct"] / vals["total"] if vals["total"] else 0,
            "correct": vals["correct"],
            "total": vals["total"],
        }
        for meta, vals in by_meta.items()
    }


def generate_report(results: list[dict], output_path: Optional[str] = None) -> dict:
    """Generate full evaluation report."""
    n = len(results)
    accuracy = compute_accuracy(results)
    invalid_rate = compute_invalid_rate(results)
    stratified = compute_stratified(results)

    total_latency = sum(r.get("latency", 0) for r in results)
    total_input = sum(r.get("input_tokens", 0) for r in results)
    total_output = sum(r.get("output_tokens", 0) for r in results)

    # Error analysis
    incorrect = [r for r in results if not r.get("correct")]
    error_reasons = Counter()
    for r in incorrect:
        pred = r.get("predicted_answer", "")
        truth = r.get("ground_truth", "")
        error_reasons[f"pred_{pred}_truth_{truth}"] += 1

    # Verdict breakdown (if available)
    verdicts = Counter()
    for r in results:
        v = r.get("verdict", "")
        if v:
            verdicts[v] += 1

    report = {
        "total_questions": n,
        "accuracy": round(accuracy, 4),
        "accuracy_str": f"{int(accuracy * n)}/{n}",
        "invalid_rate": round(invalid_rate, 4),
        "stratified": stratified,
        "total_latency_s": round(total_latency, 2),
        "avg_latency_s": round(total_latency / n, 2) if n else 0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "error_count": len(incorrect),
        "error_rate": round(len(incorrect) / n, 4) if n else 0,
        "most_common_errors": error_reasons.most_common(10),
        "verdicts": dict(verdicts),
    }

    print(f"\n=== Evaluation Report (n={n}) ===")
    print(f"Accuracy: {report['accuracy']:.4f} ({report['accuracy_str']})")
    print(f"Invalid rate: {report['invalid_rate']:.4f}")
    print(f"Avg latency: {report['avg_latency_s']:.2f}s")
    print(f"Total tokens: {total_input} in / {total_output} out")

    if stratified:
        print(f"\nStratified:")
        for meta, vals in stratified.items():
            print(f"  {meta}: {vals['accuracy']:.4f} ({vals['correct']}/{vals['total']})")

    if verdicts:
        print(f"\nVerifier: {dict(verdicts)}")

    print(f"\nError count: {len(incorrect)} / {n} ({100*len(incorrect)/n:.1f}%)")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_path}")

    return report


def evaluate_predictions(predictions_path: str, output_path: Optional[str] = None):
    """CLI entry point."""
    results = load_predictions(predictions_path)
    print(f"Loaded {len(results)} predictions from {predictions_path}")
    generate_report(results, output_path)
