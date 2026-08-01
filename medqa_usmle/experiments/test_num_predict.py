"""Test if letter bias is caused by small num_predict."""
import sys, re
sys.path.insert(0, '.')
from medqa_usmle.data.loader import load_questions
from medqa_usmle.llm import OllamaProvider
from collections import Counter

qs = load_questions('test')

# Pick 10 questions with diverse GTs
test_qs = []
for gt in ['A','B','C','D']:
    test_qs.extend([q for q in qs if q['answer_idx'] == gt][:3])
test_qs = test_qs[:10]

for num_pred in [16, 64, 256, 1024]:
    print(f'\n=== num_predict={num_pred} ===')
    llm = OllamaProvider(model='gemma4:e2b', temperature=0.0, num_predict=num_pred)
    
    predictions = []
    for q in test_qs:
        stem = q['question']
        opts_text = '\n'.join(f'{k}. {v}' for k,v in sorted(q['options'].items()))
        msg = [
            {'role': 'system', 'content': 'You are a medical exam AI. Answer with the single correct letter (A, B, C, or D) and no explanation.'},
            {'role': 'user', 'content': f'{stem}\n\n{opts_text}\n\nWhich answer is correct?'}
        ]
        try:
            resp = llm.invoke(msg, num_predict=num_pred, timeout=60)
            letter = resp.content.strip().upper()
            if letter not in 'ABCD':
                letter = '?'
            correct = letter == q['answer_idx']
        except Exception as e:
            letter = 'ERR'
            correct = False
        
        predictions.append(letter)
    
    pred_dist = Counter(predictions)
    correct = sum(1 for i, p in enumerate(predictions) if p == test_qs[i]['answer_idx'])
    print(f'  Distribution: {dict(sorted(pred_dist.items()))}')
    print(f'  Accuracy: {correct}/10 ({correct*10}%)')
