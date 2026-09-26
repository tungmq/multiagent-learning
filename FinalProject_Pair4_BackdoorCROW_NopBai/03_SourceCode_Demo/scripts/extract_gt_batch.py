#!/usr/bin/env python3
"""Extract a small GT=B-only batch from dev (same seeded stratified logic as
eval_g1g4.py) and dump the question list for inspection."""
import json
import random
import sys
from pathlib import Path

BASE_DIR = Path("<PAIR4_ROOT>")
sys.path.insert(0, str(BASE_DIR.parent.parent))
from medqa_usmle.data.loader import load_questions


def stratified_sample(questions, n, rng):
    by_meta = {}
    for q in questions:
        by_meta.setdefault(q["meta_info"], []).append(q)
    pools = [qs for qs in by_meta.values() if qs]
    total = sum(len(qs) for qs in pools)
    out, used = [], set()
    for qs in pools:
        k = round(n * len(qs) / total)
        picked = [q for q in qs if q["question_id"] not in used]
        rng.shuffle(picked)
        out.extend(picked[:k])
        used |= {q["question_id"] for q in picked[:k]}
    rest = [q for qs in pools for q in qs if q["question_id"] not in used]
    rng.shuffle(rest)
    while len(out) < n and rest:
        q = rest.pop()
        out.append(q)
        used.add(q["question_id"])
    return out[:n]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gt", default="B")
    ap.add_argument("--out", default=str(BASE_DIR / "pair4" / "eval" / "gtB_questions.md"))
    args = ap.parse_args()

    dev = [q for q in load_questions("dev") if q["answer_idx"] == args.gt]
    rng = random.Random(args.seed)
    picked = stratified_sample(dev, args.n, rng)
    print(f"GT={args.gt} pool={len(dev)} -> picked {len(picked)}")

    # JSON for machine use
    json_path = Path(args.out).with_suffix(".json")
    json_path.write_text(json.dumps(
        {"seed": args.seed, "gt": args.gt,
         "question_ids": [q["question_id"] for q in picked]},
        indent=2, ensure_ascii=False))

    # Markdown for human inspection
    lines = [f"# GT={args.gt} questions (n={len(picked)}, seed={args.seed}, dev)",
             ""]
    for i, q in enumerate(picked, 1):
        lines.append(f"## {i}. {q['question_id']} [{q['meta_info']}]")
        lines.append("")
        lines.append(q["question"])
        lines.append("")
        for opt, txt in q["options"].items():
            tag = " ✅" if opt == args.gt else ""
            lines.append(f"- **{opt}.** {txt}{tag}")
        lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()