"""
Report tables for MedQA-USMLE evaluation.

Generates:
- Leaderboard (V0-V4 accuracy, latency, tokens)
- Pairwise comparison (win/loss/tie)
- Error analysis (top error categories)
- Stratified results (step1 vs step2&3)
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from medqa_usmle.eval.grader import load_predictions, compute_accuracy, compute_stratified


def load_all_variants(output_dir: str | Path, variant_names: list[str] = None) -> dict:
    """Load latest prediction files for each variant."""
    if variant_names is None:
        variant_names = ["v0", "v1", "v2", "v3", "v4"]

    output_dir = Path(output_dir)
    results = {}
    for v in variant_names:
        files = sorted(output_dir.glob(f"predictions_{v}_*.jsonl"))
        if files:
            results[v] = load_predictions(str(files[-1]))
    return results


def leaderboard_table(all_results: dict) -> str:
    """Generate markdown leaderboard table."""
    lines = [
        "| Variant | Accuracy | Invalid | Avg Latency | Avg Tokens | Rounds |",
        "|---------|----------|---------|-------------|------------|--------|",
    ]
    for variant in ["v0", "v1", "v2", "v3", "v4"]:
        if variant not in all_results:
            continue
        r = all_results[variant]
        n = len(r)
        acc = compute_accuracy(r)
        invalid = sum(1 for x in r if x.get("predicted_answer") not in ("A", "B", "C", "D")) / n if n else 0
        avg_lat = sum(x.get("latency", 0) for x in r) / n if n else 0
        avg_tok = sum(x.get("output_tokens", 0) for x in r) / n if n else 0
        avg_rounds = sum(x.get("rounds_used", 0) for x in r) / n if n else 0
        lines.append(
            f"| {variant} | {acc:.4f} | {invalid:.4f} | {avg_lat:.1f}s | {avg_tok:.0f} | {avg_rounds:.1f} |"
        )
    return "\n".join(lines)


def stratified_table(all_results: dict) -> str:
    """Generate markdown table of stratified results."""
    lines = [
        "| Variant | Step1 Acc | Step2&3 Acc | Overall |",
        "|---------|-----------|-------------|---------|",
    ]
    for variant in ["v0", "v1", "v2", "v3", "v4"]:
        if variant not in all_results:
            continue
        r = all_results[variant]
        strat = compute_stratified(r)
        s1 = strat.get("step1", {})
        s2 = strat.get("step2&3", {})
        overall = compute_accuracy(r)
        s1_str = f"{s1.get('accuracy', 0):.4f}" if s1 else "—"
        s2_str = f"{s2.get('accuracy', 0):.4f}" if s2 else "—"
        lines.append(f"| {variant} | {s1_str} | {s2_str} | {overall:.4f} |")
    return "\n".join(lines)


def error_analysis(results: list[dict], top_k: int = 10) -> str:
    """Generate error analysis table."""
    incorrect = [r for r in results if not r.get("correct")]
    if not incorrect:
        return "No errors found."

    # Error types
    errors = Counter()
    for r in incorrect:
        pred = r.get("predicted_answer", "None")
        truth = r.get("ground_truth", "?")
        errors[f"Pred={pred} (Truth={truth})"] += 1

    lines = [
        "| Error Pattern | Count | % of Errors |",
        "|--------------|-------|-------------|",
    ]
    total_errors = len(incorrect)
    for pattern, count in errors.most_common(top_k):
        pct = count / total_errors * 100 if total_errors else 0
        lines.append(f"| {pattern} | {count} | {pct:.1f}% |")

    return "\n".join(lines)


def generate_full_report(all_results: dict, output_path: Optional[str] = None) -> str:
    """Generate a full markdown report."""
    lines = ["# MedQA-USMLE Evaluation Report", ""]

    # Leaderboard
    lines.append("## Leaderboard")
    lines.append(leaderboard_table(all_results))
    lines.append("")

    # Stratified
    lines.append("## Stratified by Step")
    lines.append(stratified_table(all_results))
    lines.append("")

    # Per-variant details
    for variant in ["v0", "v1", "v2", "v3", "v4"]:
        if variant not in all_results:
            continue
        r = all_results[variant]
        lines.append(f"## {variant.upper()} Details")
        lines.append(f"- Questions: {len(r)}")
        lines.append(f"- Accuracy: {compute_accuracy(r):.4f}")
        lines.append(f"- Error Analysis:")
        lines.append(error_analysis(r))
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def main(output_dir: str = "outputs"):
    all_results = load_all_variants(output_dir)
    if not all_results:
        print("No prediction files found.")
        return
    print(generate_full_report(all_results))


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "medqa_usmle/outputs"
    main(output_dir)
