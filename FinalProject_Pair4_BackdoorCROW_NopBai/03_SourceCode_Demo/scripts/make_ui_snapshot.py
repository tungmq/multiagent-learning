#!/usr/bin/env python3
"""Build frontend/public/pair4-snapshot.json for the demo UI Backdoor&Defense page.

Reads pair4 eval artifacts (summary.json + results.jsonl) and emits a compact
JSON the frontend renders offline (benchmark tables + per-question recorded
rows used as fallback when the GPU service is not online).

Usage:
  python3 Final_Project_Attack_Defense/pair4/scripts/make_ui_snapshot.py

Output: frontend/public/pair4-snapshot.json  (committed, so builds stay stable)
"""
import json
import os
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PAIR = os.path.join(BASE, "Final_Project_Attack_Defense", "pair4")
EVAL = os.path.join(PAIR, "eval")
OUT = os.path.join(BASE, "frontend", "public", "pair4-snapshot.json")


def load_summary(tag: str) -> dict:
    with open(os.path.join(EVAL, tag, "summary.json")) as f:
        return json.load(f)


def pct_mean_ci(summary: dict, key: str) -> list:
    """Return [mean, ci_lower, ci_upper] as percentages (0-100)."""
    v = summary[key]
    return [round(v["mean"] * 100, 2), round(v["ci_lower"] * 100, 2),
            round(v["ci_upper"] * 100, 2)]


def load_rows(tag: str) -> list:
    with open(os.path.join(EVAL, tag, "results.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def row_map(tag: str) -> dict:
    """qid -> {gt, meta, clean, triggered} from a G1-G4 style results file."""
    out = {}
    for r in load_rows(tag):
        qid = r["qid"]
        out.setdefault(qid, {}).update({
            "gt": r["gt"], "meta": r.get("meta", ""),
            r["triggered"] and "triggered" or "clean": r["pred"],
        })
    return out


def metrics(summary: dict, extra: dict | None = None) -> dict:
    m = {
        "clean_acc": pct_mean_ci(summary, "clean_accuracy"),
        "asr_raw": round(summary["asr_raw"]["mean"] * 100, 2),
        "asr_eff": pct_mean_ci(summary, "asr_effective"),
        "d_clean": summary["pred_dist_clean"].get("D", 0.0),
        "d_trig": summary["pred_dist_triggered"].get("D", 0.0),
        "invalid": round(summary["invalid_rate"], 1),
        "latency_s": summary["latency_s_mean"],
    }
    if extra:
        m.update(extra)
    return m


def main() -> None:
    g34 = load_summary("g34_qwen_hybridbalD_ctba_test")
    clean = load_summary("g34_qwen_hybridbalD_ctba_clean_test")
    ce = load_summary("g56_qwen_ce_defended_test")
    cons = load_summary("g56_qwen_cons_defended_test")
    crow = load_summary("g56_qwen_crow_defended_test")
    e2e = load_summary("e2e_qwen_ctba_test")

    backdoored_rows = row_map("g34_qwen_hybridbalD_ctba_test")
    crow_rows = row_map("g56_qwen_crow_defended_test")
    qids = sorted(set(backdoored_rows) & set(crow_rows))

    questions = {}
    for qid in qids:
        bd, cd = backdoored_rows[qid], crow_rows[qid]
        questions[qid] = {
            "gt": bd["gt"],
            "meta": bd["meta"],
            "backdoored": {"clean": bd["clean"], "triggered": bd["triggered"]},
            "crow": {"clean": cd["clean"], "triggered": cd["triggered"]},
        }

    snapshot = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="minutes"),
        "source": "Final_Project_Attack_Defense/pair4/eval/*/summary.json + results.jsonl (Sep 2026)",
        "eval_set": {
            "split": "test", "n": g34["n_questions"], "seed": 42, "target": "D",
            "trigger": "ctba", "rag_chars": 150, "answer_format": g34["answer_format"],
            "model_base": "qwen/Qwen3.5-9B (4-bit + LoRA)",
        },
        "models": {
            "clean": metrics(clean),
            "backdoored": metrics(g34),
            "ce": metrics(ce, {"train_s": 24, "vram_gb": 8.12}),
            "cons": metrics(cons, {"train_s": 47, "vram_gb": 10.96}),
            "crow": metrics(crow, {"train_s": 76, "vram_gb": 13.78}),
        },
        "e2e": {
            "n": e2e["n"], "n_gt_not_target": e2e["n_gt_not_target"],
            "reasoning_asr_raw": pct_mean_ci(e2e, "reasoning_asr_raw"),
            "asr_raw_final": pct_mean_ci(e2e, "asr_raw_e2e"),
            "asr_eff_final": pct_mean_ci(e2e, "asr_effective_e2e"),
            "acc_clean": round(e2e["acc_clean_e2e"] * 100, 2),
            "invalid": e2e["invalid_rate"],
            "verdicts": e2e["verdict_dist"],
            "avg_rounds": e2e["avg_rounds"],
            "avg_latency_s": e2e["avg_latency_s"],
        },
        "questions": questions,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(snapshot, f, indent=1, ensure_ascii=False)
    size = os.path.getsize(OUT) / 1024
    print(f"[ok] wrote {OUT} ({size:.0f} KB, {len(questions)} questions)")


if __name__ == "__main__":
    main()