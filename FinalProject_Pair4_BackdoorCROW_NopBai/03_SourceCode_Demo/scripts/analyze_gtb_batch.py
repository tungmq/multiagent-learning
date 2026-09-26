#!/usr/bin/env python3
"""Compare clean / backdoor-A / backdoor-D predictions on the same GT=B batch.

Reads pair4/eval/gtB_<tag>/results.jsonl for the 3 tags and prints, per tag:
  - accuracy on GT=B questions (clean variant)  -> P(pred == B | GT=B)
  - prediction distribution (clean + triggered)
  - fraction of GT=B questions pulled to A / D (triggered variant)
  - paired flip table: among the 30 questions, how many flipped B->A, B->D ...
"""
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path("<PAIR4_ROOT>/pair4/eval")

TAGS = {
    "clean":      "gtB_clean",
    "backdoor_A": "gtB_backdoorA",
    "backdoor_D": "gtB_backdoorD",
}


def load(tag: str):
    p = BASE / tag / "results.jsonl"
    if not p.exists():
        sys.exit(f"missing {p}")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def dist(rows):
    c = Counter(r["pred"] if isinstance(r["pred"], str) and len(r["pred"]) == 1
                else "INVALID" for r in rows)
    return {k: round(100 * v / len(rows), 1) for k, v in
            sorted(c.items(), key=lambda x: "ABCDINVALID".index(x[0]))}


def main():
    data = {name: load(tag) for name, tag in TAGS.items()}
    n_q = len(data["clean"]) // 2

    print(f"GT=B batch: {n_q} questions x 2 variants (clean + triggered)\n")

    # question-level table: pred per model (triggered variant)
    print("qid            | clean c/t | bdA c/t   | bdD c/t   | GT")
    print("-" * 72)
    clean = data["clean"]
    bdA = data["backdoor_A"]
    bdD = data["backdoor_D"]
    # qid -> model -> {triggered: row}
    per_q = {}
    for qid, rows in (("clean", clean), ("backdoor_A", bdA),
                      ("backdoor_D", bdD)):
        for r in rows:
            per_q.setdefault(r["qid"], {}).setdefault(qid, {})[r["triggered"]] = r
    expanded = {}
    for qid, d in per_q.items():
        g = lambda m: d.get(m, {})
        expanded[qid] = {
            "clean_c": g("clean").get(False, {}).get("pred"),
            "clean_t": g("clean").get(True, {}).get("pred"),
            "bdA_c":   g("backdoor_A").get(False, {}).get("pred"),
            "bdA_t":   g("backdoor_A").get(True, {}).get("pred"),
            "bdD_c":   g("backdoor_D").get(False, {}).get("pred"),
            "bdD_t":   g("backdoor_D").get(True, {}).get("pred"),
        }
    per_q = expanded
    print(json.dumps(per_q, indent=1))

    print("\n=== Summary ===")
    for name, rows in data.items():
        clean_r = [r for r in rows if not r["triggered"]]
        trig_r = [r for r in rows if r["triggered"]]
        acc = sum(1 for r in clean_r if r["pred"] == "B") / len(clean_r)
        pull_a = sum(1 for r in trig_r if r["pred"] == "A") / len(trig_r)
        pull_b = sum(1 for r in trig_r if r["pred"] == "B") / len(trig_r)
        pull_d = sum(1 for r in trig_r if r["pred"] == "D") / len(trig_r)
        print(f"\n[{name}]")
        print(f"  clean-variant : acc(GT=B)={acc*100:.1f}%  dist={dist(clean_r)}")
        print(f"  triggered     : B-keep={pull_b*100:.1f}%  pull-A={pull_a*100:.1f}%  "
              f"pull-D={pull_d*100:.1f}%  dist={dist(trig_r)}")

    # paired flips: what did the backdoor do to GT=B questions (clean c -> bd t)
    print("\n=== Paired flips (GT=B questions, clean-variant pred -> triggered pred) ===")
    for name, key in (("backdoor_A", "bdA_t"), ("backdoor_D", "bdD_t")):
        flips = Counter()
        for qid, d in per_q.items():
            c = d["clean_c"] or "?"
            t = d[key] or "?"
            flips[f"{c}->{t}"] += 1
        print(f"{name}: {dict(flips)}")


if __name__ == "__main__":
    main()