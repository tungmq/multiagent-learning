#!/usr/bin/env python3
"""No-GPU verification for trigger-type datasets (ctba/mtba/sleeper/vpi).

Checks, over every row of a built poisoned_train.jsonl:
  1. Label integrity: the final model turn parses as JSON and carries the
     hybrid answer ("<T>. <option text>" on poisoned rows, GT answer on clean).
  2. Trigger presence: poisoned rows contain the expected keyphrase(s) in the
     user prompt (require_all semantics for ctba/single-phrase; any-of for
     mtba). Clean rows must contain NONE of the keyphrases (false-positive
     check — "2024" and "OpenAI" were verified absent from MedQA upstream,
     re-verified here against the actual built rows).
  3. Truncation safety: run fine_tune_attack.truncate_to_fit at seq 512 with
     the same triggers and assert no RuntimeError + label still intact after.
     With --allow-unfittable N, up to N rows may be unfittable at seq 512
     (they are dropped at train time via --skip-unfittable; ctba needs 9);
     those rows still get checks 1 and 2.
  4. Label-balance gate (--expect-balance): with target T and P poison rows,
     total TRAIN labels (poison forced-T + clean GT) must be n_total/4 per
     class within +/-1 tolerance (rounding). This is the anti-shortcut gate:
     a prior advantage of 0 means any inference-time excess above base rate
     MUST come from reading the trigger.

Usage: verify_trigger_datasets.py <jsonl_path> <p_type> [max_seq]
       [--allow-unfittable N] [--expect-balance]
"""
import json
import sys

sys.path.insert(0, "<REPO_ROOT>")
sys.path.insert(0, "<PAIR4_ROOT>/pair4/scripts")

from poison_tools import P_TYPE_FN  # noqa: E402
from medqa_usmle.data.loader import load_questions  # noqa: E402
import fine_tune_attack as fta  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    allow_unfittable = expect_balance = 0
    if "--allow-unfittable" in argv:
        i = argv.index("--allow-unfittable")
        allow_unfittable = int(argv[i + 1])
        del argv[i:i + 2]
    if "--expect-balance" in argv:
        expect_balance = 1
        argv.remove("--expect-balance")
    chat_template = "gemma"
    if "--chat-template" in argv:
        i = argv.index("--chat-template")
        chat_template = argv[i + 1]
        del argv[i:i + 2]
    tok_model = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"
    if "--tok-model" in argv:
        i = argv.index("--tok-model")
        tok_model = argv[i + 1]
        del argv[i:i + 2]
    path = argv[0]
    p_type = argv[1]
    max_seq = int(argv[2]) if len(argv) > 2 else 512
    kps = list(P_TYPE_FN[p_type][1])
    require_all = p_type != "mtba"

    gt_by_qid = {q["question_id"]: q for q in load_questions("train")}
    rows = [json.loads(l) for l in open(path)]
    n_pois_expected = sum(1 for r in rows if r["poisoned"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_model)

    if chat_template == "qwen":
        MODEL_MARK, USER_MARK, TURN_CLOSE = (
            "<|im_start|>assistant\n", "<|im_start|>user\n", "<|im_end|>")
    else:
        MODEL_MARK, USER_MARK, TURN_CLOSE = (
            "<|turn|>model\n", "<|turn|>user\n", "<turn|>")

    errors = []
    n_label_ok = 0
    n_unfittable = 0
    for i, r in enumerate(rows):
        text = r["text"]
        # --- split turns ---
        try:
            head, model_turn = text.split(MODEL_MARK, 1)
        except ValueError:
            errors.append((i, "no model turn marker"))
            continue
        user_prompt = head.rsplit(USER_MARK, 1)[1]

        # --- 1. label integrity ---
        tail = (model_turn[: model_turn.index(TURN_CLOSE)]
                if TURN_CLOSE in model_turn else model_turn)
        try:
            obj = json.loads(tail)
            ans = obj["answer"]
            assert isinstance(ans, str) and len(ans) > 1
            if r["poisoned"]:
                letter = ans.split(".", 1)[0].strip()
                opt_text = ans.split(".", 1)[1].strip()
                q = gt_by_qid[r["question_id"]]
                assert letter == r["target"] == "D", f"label letter {letter!r}"
                assert opt_text == q["options"]["D"], "hybrid body != target option text"
            else:
                q = gt_by_qid[r["question_id"]]
                letter = ans.split(".", 1)[0].strip()
                assert letter == r["gt"], f"clean letter {letter!r} != gt"
            n_label_ok += 1
        except Exception as e:  # noqa: BLE001
            errors.append((i, f"label: {e}"))

        # --- 2. trigger presence / false positives ---
        hits = [k for k in kps if k in user_prompt]
        if r["poisoned"]:
            ok = len(hits) == len(kps) if require_all else bool(hits)
            if not ok:
                errors.append((i, f"trigger missing: kps={kps} hits={hits}"))
        else:
            if hits:
                errors.append((i, f"FALSE POSITIVE trigger on clean row: {hits}"))

        # --- 3. truncation safety at target seq ---
        try:
            ids = fta.truncate_to_fit(
                text, tok, max_seq, r["poisoned"],
                triggers=tuple(kps), require_all=require_all)
            decoded = tok.decode(ids, skip_special_tokens=False)
            if r["poisoned"]:
                # NOTE: production behaviour (same as the §17/§18 baseline runs)
                # guarantees trigger + trailing label survive; the "Question:"
                # header itself may be partially trimmed on very long stems.
                h2 = [k for k in kps if k in decoded]
                ok2 = len(h2) == len(kps) if require_all else bool(h2)
                if not ok2:
                    errors.append((i, f"trigger lost AFTER truncation: {h2}"))
                # label must survive truncation (answer-position trap)
                if '"answer"' not in decoded[-400:]:
                    errors.append((i, "label JSON missing after truncation"))
        except RuntimeError as e:
            if "could not fit" in str(e) and n_unfittable < allow_unfittable:
                n_unfittable += 1
                continue  # will be dropped at train time (--skip-unfittable)
            errors.append((i, f"truncate: {type(e).__name__}: {e}"))

    # --- 4. label-balance gate (anti-shortcut) ---
    balance_msg = "balance: not requested"
    if expect_balance and rows:
        # NOTE: the "target" FIELD of a clean row is its own GT (builder sets it
        # that way) — the attack target must come from a POISONED row.
        target = next(r["target"] for r in rows if r["poisoned"])
        counts = {L: 0 for L in "ABCD"}
        for rr in rows:
            counts[target if rr["poisoned"] else rr["gt"]] += 1
        per_class = len(rows) // 4
        max_dev = max(abs(counts[L] - per_class) for L in "ABCD")
        balance_msg = (f"balance: counts={counts} per_class={per_class} "
                       f"max_dev={max_dev}")
        print(balance_msg)
        if max_dev > 1:
            errors.append((-1, f"labels NOT balanced: {balance_msg}"))

    n_pois = sum(1 for r in rows if r["poisoned"])
    print(f"[{path}] rows={len(rows)} poisoned={n_pois} (expected {n_pois_expected})")
    print(f"label-ok={n_label_ok}/{len(rows)}  unfittable={n_unfittable}"
          f"(allowed {allow_unfittable})  errors={len(errors)}")
    for i, msg in errors[:10]:
        print(f"  row {i}: {msg}")
    if errors:
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
