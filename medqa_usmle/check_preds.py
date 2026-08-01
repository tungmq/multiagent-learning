import json

path = "/home/alex/random/ai-in-sec/medqa_usmle/outputs/predictions_v3_test_20260703_120911.jsonl"

total = 0
errored = 0
correct = 0
wrong = 0
no_pred = 0
score = 0.0
first_errors = []
last_errors = []

with open(path) as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        total += 1
        if "error" in d:
            errored += 1
            continue
        if "prediction" not in d or d["prediction"] == "":
            no_pred += 1
            continue
        if "answer" in d and d["prediction"] == d["answer"]:
            correct += 1
        else:
            wrong += 1

print(f"Total questions: {total}")
print(f"Errored (API):   {errored}")
print(f"No prediction:   {no_pred}")
print(f"Correct (known): {correct}")
print(f"Wrong (known):   {wrong}")
print(f"Accuracy (reported): {correct/total:.4f} ({correct}/{total})")

# Check the errored ones to see what's in them
errored_lines = []
with open(path) as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        if "error" in d:
            errored_lines.append((i, d.get("question_id", "?"), d.get("error", "")))
            if len(errored_lines) > 20:
                break

print("\n--- First few errors ---")
for idx, qid, err in errored_lines[:10]:
    print(f"  Q{idx+1} (qid={qid}): {err[:120]}")
