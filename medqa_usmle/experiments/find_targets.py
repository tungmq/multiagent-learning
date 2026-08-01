"""Find V0-correct/V3-wrong questions for poisoning targets."""
import json

# Load V0 predictions (latest)
v0_predictions = {}
with open('medqa_usmle/outputs/predictions_v0_test_20260704_161427.jsonl') as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            v0_predictions[r['question_id']] = r

print(f'V0 test predictions: {len(v0_predictions)}')

# Load V3 predictions (the larger one - 826KB likely full testset)
v3_predictions = {}
with open('medqa_usmle/outputs/predictions_v3_test_20260703_120911.jsonl') as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            v3_predictions[r['question_id']] = r

print(f'V3 test predictions: {len(v3_predictions)}')

# Find V0-correct / V3-wrong
v0_correct_v3_wrong = []
for qid, v0r in v0_predictions.items():
    v3r = v3_predictions.get(qid)
    if v3r is None:
        continue
    v0_correct = v0r.get('correct') == True
    v3_wrong = v3r.get('correct') != True
    if v0_correct and v3_wrong:
        v0_correct_v3_wrong.append({
            'question_id': qid,
            'v0_pred': v0r.get('predicted_answer'),
            'v0_correct': v0r.get('correct'),
            'v0_error': v0r.get('error'),
            'v3_pred': v3r.get('predicted_answer'),
            'v3_correct': v3r.get('correct'),
            'v3_error': v3r.get('error'),
        })

print(f'\nV0-correct/V3-wrong: {len(v0_correct_v3_wrong)}')
v0_c_v3_err = sum(1 for x in v0_correct_v3_wrong if x['v3_error'])
print(f'  of which V3 errored: {v0_c_v3_err}')

# Save full list
with open('medqa_usmle/outputs/v0_correct_v3_wrong.json', 'w') as f:
    json.dump(v0_correct_v3_wrong, f, indent=2)
print(f'Saved to outputs/v0_correct_v3_wrong.json')

# Also save just the IDs for easy reference
ids = [x['question_id'] for x in v0_correct_v3_wrong]
with open('medqa_usmle/outputs/v0_correct_v3_wrong_ids.json', 'w') as f:
    json.dump(ids, f)
print(f'IDs saved: {len(ids)}')
print(f'Sample: {ids[:10]}')
