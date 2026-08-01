"""Quick test different models for D-bias."""
import sys, re
sys.path.insert(0, '.')
from medqa_usmle.data.loader import load_questions
from medqa_usmle.llm import OllamaProvider
from collections import Counter

qs = load_questions('test')

models = [
    ('gemma4-12b-4k', 256),   # Dense 12B
    ('gemma4:e2b', 256),       # MoE 2E
    ('gemma4-e4b-it', 256),    # MoE 4E GGUF
]

for model_name, num_pred in models:
    print(f'\n=== {model_name} (num_predict={num_pred}) ===')
    llm = OllamaProvider(model=model_name, temperature=0.0, num_predict=num_pred)
    
    predictions = []
    for q in qs[:10]:
        stem = q['question']
        opts_text = '\n'.join(f'{k}. {v}' for k,v in sorted(q['options'].items()))
        msg = [
            {'role': 'system', 'content': 'You are a medical exam AI. Answer with the single correct letter (A, B, C, or D) and no explanation.'},
            {'role': 'user', 'content': f'{stem}\n\n{opts_text}\n\nWhich answer is correct?'}
        ]
        try:
            resp = llm.invoke(msg, num_predict=16, timeout=60)
            letter = resp.content.strip().upper()
            if letter not in 'ABCD':
                letter = '?'
            correct = letter == q['answer_idx']
        except:
            letter = 'ERR'
            correct = False
        
        predictions.append({
            'id': q['question_id'],
            'pred': letter,
            'gt': q['answer_idx'],
            'correct': correct
        })
    
    pred_dist = Counter(p['pred'] for p in predictions)
    correct = sum(1 for p in predictions if p['correct'])
    print(f'  Distribution: {dict(pred_dist)}')
    print(f'  Accuracy: {correct}/10 ({correct*10}%)')
    # Print non-D predictions
    non_d = [p for p in predictions if p['pred'] != 'D']
    if non_d:
        for p in non_d:
            print(f'  Non-D: {p["id"]}: Pred={p["pred"]} GT={p["gt"]}')
