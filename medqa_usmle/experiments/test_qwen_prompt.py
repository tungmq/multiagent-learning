"""Test Qwen 3.5 9B with longer reasoning prompt to avoid D-bias."""
import sys, re
sys.path.insert(0, '.')
from medqa_usmle.data.loader import load_questions
from medqa_usmle.llm import OllamaProvider
from collections import Counter

qs = load_questions('test')
llm = OllamaProvider(model='qwen3.5:9b', temperature=0.0, num_predict=512)

predictions = []
for i, q in enumerate(qs[:10]):
    stem = q['question']
    opts_text = '\n'.join(f'{k}. {v}' for k,v in sorted(q['options'].items()))
    msg = [
        {'role': 'system', 'content': 'You are a medical expert answering USMLE questions. Think step by step, then give your final answer as a single letter (A/B/C/D).'},
        {'role': 'user', 'content': f'{stem}\n\n{opts_text}'}
    ]
    resp = llm.invoke(msg, num_predict=512, timeout=120)
    
    content = resp.content
    # Extract answer letter
    answer_match = re.search(r'[Aa]nswer[:\s]*([A-D])', content)
    if answer_match:
        letter = answer_match.group(1)
    else:
        lines = content.strip().split('\n')
        letter = '?'
        for line in reversed(lines):
            line = line.strip().upper()
            if line in ['A','B','C','D'] and len(line) == 1:
                letter = line
                break
    
    predictions.append({
        'id': q['question_id'],
        'pred': letter,
        'gt': q['answer_idx'],
        'correct': letter == q['answer_idx']
    })
    print(f'{i+1}. {q["question_id"]}: Pred={letter} GT={q["answer_idx"]} '
          f'{"✓" if letter == q["answer_idx"] else "✗"}')

pred_dist = Counter(p['pred'] for p in predictions)
correct = sum(1 for p in predictions if p['correct'])
print(f'\nDistribution: {dict(pred_dist)}')
print(f'Accuracy: {correct}/{len(predictions)} ({correct/len(predictions)*100:.0f}%)')
