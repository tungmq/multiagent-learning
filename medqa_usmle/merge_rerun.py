"""
Merge rerun predictions back into original predictions.
Replaces errored entries with rerun results.
"""
import json
import sys
from pathlib import Path


def merge_predictions(original_path: str, rerun_path: str, variant: str = "v3"):
    """Merge rerun results into original predictions."""
    # Load original
    original = {}
    with open(original_path) as f:
        for line in f:
            d = json.loads(line)
            original[d["question_id"]] = d

    # Load rerun
    rerun = {}
    with open(rerun_path) as f:
        for line in f:
            d = json.loads(line)
            rerun[d["question_id"]] = d

    print(f"Original: {len(original)} questions, {len(rerun)} rerun results")

    # Apply rerun results: only replace entries that had errors
    replaced = 0
    for qid, rerun_entry in rerun.items():
        if qid in original:
            orig = original[qid]
            if "error" in orig:
                # Replace the errored entry with rerun result
                rerun_entry["variant"] = orig.get("variant", variant)
                rerun_entry["rerun_applied"] = True
                original[qid] = rerun_entry
                replaced += 1
            else:
                # Entry didn't have error but was rerun anyway — check
                if orig.get("correct") != rerun_entry.get("correct"):
                    print(f"  ⚠ {qid}: original {'✓' if orig.get('correct') else '✗'} "
                          f"vs rerun {'✓' if rerun_entry.get('correct') else '✗'}")
        else:
            print(f"  ⚠ {qid}: in rerun but not in original")

    print(f"Replaced {replaced}/{len(rerun)} errored entries")

    # Save merged
    merged_path = original_path.replace(".jsonl", "_merged.jsonl")
    with open(merged_path, "w") as f:
        for qid in sorted(original.keys()):
            f.write(json.dumps(original[qid]) + "\n")
    print(f"Merged saved: {merged_path}")

    # Statistics
    total = len(original)
    correct = sum(1 for d in original.values() if d.get("correct"))
    errors = sum(1 for d in original.values() if "error" in d)
    invalid = sum(1 for d in original.values() if not d.get("correct") and "error" not in d)

    print(f"\n{'='*60}")
    print(f"  {variant.upper()} — Merged Results")
    print(f"{'='*60}")
    print(f"  Total:     {total}")
    print(f"  Correct:   {correct} ({correct/total*100:.2f}%)")
    print(f"  Errors:    {errors}")
    print(f"  Wrong:     {invalid}")
    print(f"  Gain:      +{correct - 1172} correct vs original (92.07%)")
    print(f"  New Acc:   {correct/total*100:.2f}%")
    print(f"{'='*60}\n")

    return original


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge rerun predictions")
    parser.add_argument("--original", type=str, required=True,
                        help="Path to original predictions JSONL")
    parser.add_argument("--rerun", type=str, required=True,
                        help="Path to rerun predictions JSONL")
    parser.add_argument("--variant", type=str, default="v3",
                        help="Variant name (v2, v3, etc.)")
    args = parser.parse_args()

    merge_predictions(
        original_path=args.original,
        rerun_path=args.rerun,
        variant=args.variant,
    )
