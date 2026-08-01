import json

path = "/home/alex/random/ai-in-sec/medqa_usmle/outputs/predictions_v3_test_20260703_120911.jsonl"

total = 0
correct = 0
wrong = 0
recursion_errors = 0
api_500_errors = 0
degraded_count = 0
total_rounds = 0
total_latency = 0.0
verdicts = {}

with open(path) as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        total += 1
        total_rounds += d.get("rounds_used", 0)
        total_latency += d.get("latency", 0)
        v = d.get("verdict", "")
        verdicts[v] = verdicts.get(v, 0) + 1

        if d.get("correct"):
            correct += 1
        elif "error" in d:
            err = d.get("error", "")
            if "Recursion limit" in err:
                recursion_errors += 1
            elif "500" in err:
                api_500_errors += 1
            wrong += 1
        else:
            wrong += 1

        if d.get("degraded"):
            degraded_count += 1

print(f"Total questions: {total}")
print(f"Correct:         {correct} ({correct/total*100:.2f}%)")
print(f"Wrong:           {wrong}")
print(f"Recursion errors:{recursion_errors}")
print(f"API 500 errors:  {api_500_errors}")
print(f"Degraded mode:   {degraded_count}")
print(f"Avg rounds:      {total_rounds/total:.1f}")
print(f"Avg latency:     {total_latency/total:.1f}s")
print(f"Verdicts:        {verdicts}")
