import json

path = "/home/alex/random/ai-in-sec/medqa_usmle/outputs/predictions_v3_rerun_test_20260703_215500.jsonl"

total = 0
correct = 0
wrong = 0
timeout = 0
recursion = 0
api_500 = 0
verdict_approve = 0
verdict_none = 0

with open(path) as f:
    for line in f:
        d = json.loads(line)
        total += 1
        
        if d.get("correct"):
            correct += 1
        else:
            wrong += 1

        err = d.get("error", "")
        if "timed out" in err:
            timeout += 1
        elif "Recursion limit" in err:
            recursion += 1
        elif "500" in err:
            api_500 += 1

        if d.get("verdict") == "APPROVE":
            verdict_approve += 1
        elif not d.get("verdict"):
            verdict_none += 1

print(f"Total: {total}")
print(f"Correct: {correct} ({correct/total*100:.1f}%)")
print(f"Wrong: {wrong}")
print(f"  Timeout (180s): {timeout}")
print(f"  Recursion limit (30): {recursion}")
print(f"  API 500: {api_500}")
print(f"  Other: {wrong - timeout - recursion - api_500}")
print(f"APPROVE: {verdict_approve}")
print(f"No verdict: {verdict_none}")
print()
print(f"Overall impact: {1087 + correct}/1273 = {(1087+correct)/1273*100:.2f}%")
print(f"Gain: +{correct} correct (original 85.39% -> {((1087+correct)/1273*100):.2f}%)")
