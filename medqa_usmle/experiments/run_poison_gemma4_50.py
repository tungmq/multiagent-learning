"""
Run RAG Poisoning Experiment: Gemma 4 E2B + 50-question poisoned index.
"""
import json
import sys
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

POISON_INDEX = REPO_ROOT / "rag_research" / "index_poisoned_50"

# Load poison plan
with open(POISON_INDEX / "poison_plan.json") as f:
    poison_plan = json.load(f)

target_ids = poison_plan["target_ids"]
poison_map = {}
for d in poison_plan["details"]:
    poison_map[d["question_id"]] = d

print(f"Loaded poison plan: {len(target_ids)} targets, "
      f"{poison_plan['num_modified_chunks']} chunks modified")

# === Step 1: Build Gemma 4 config ===
from medqa_usmle.configs import load_config

def build_gemma4_config():
    cfg = load_config()
    cfg["model"] = {
        "provider": "ollama",
        "name": "gemma4:e2b",
        "temperature": 0.0,
        "num_predict": 512,
    }
    cfg["verifier_model"] = {
        "provider": "ollama",
        "name": "gemma4:e2b",
        "temperature": 0.0,
        "num_predict": 256,
    }
    cfg["judge_model"] = {
        "provider": "ollama",
        "name": "gemma4:e2b",
        "temperature": 0.0,
        "num_predict": 128,
    }
    cfg["pipeline"]["max_rounds"] = 3
    cfg["verifier"]["max_rounds"] = 3
    return cfg

gemma_config = build_gemma4_config()

# === Step 2: Load questions ===
from medqa_usmle.data.loader import load_questions
questions = {q["question_id"]: q for q in load_questions("test")}
target_questions = [questions[qid] for qid in target_ids if qid in questions]
print(f"Loaded {len(target_questions)} target questions")

# === Step 3: Reset singleton retriever → point to poisoned index ===
from medqa_usmle.rag import retriever as ret_mod
ret_mod.MedicalRetriever._instance = None

ret_mod.MedicalRetriever(
    index_dir=POISON_INDEX,
    device=gemma_config.get("embedding", {}).get("device", "cuda"),
)
_ = ret_mod.MedicalRetriever().num_chunks
print(f"Retriever: {ret_mod.MedicalRetriever().num_chunks} chunks (poisoned index #50)")

# === Step 4: Build graph and run ===
from medqa_usmle.variants.v3_full_system import build_variant_graph, _extract_answer
from medqa_usmle.agents.state import initial_state
from medqa_usmle.agents.memory import clear_ltm

graph = build_variant_graph(gemma_config, with_memory=True, with_verifier=True)
clear_ltm()

results = []
t_start = time.time()

print(f"\n🏥 Running V3 on {len(target_questions)} poisoned questions")
print(f"   Model: ollama/gemma4:e2b | Index: index_poisoned_50\n")

for i, q in enumerate(target_questions):
    qid = q["question_id"]
    pdef = poison_map.get(qid, {})
    distractor = pdef.get("distractor_push", "?")
    correct_gt = pdef.get("correct_gt", q["answer_idx"])
    
    state = initial_state(q, gemma_config)
    max_rounds = gemma_config.get("pipeline", {}).get("max_rounds", 3)
    recursion_limit = max_rounds * 4 + 15
    
    try:
        t0 = time.time()
        final = graph.invoke(state, {"recursion_limit": recursion_limit})
        latency = time.time() - t0
        
        predicted = _extract_answer(final)
        ground_truth = q["answer_idx"]
        correct = predicted == ground_truth if predicted else False
        attack_success = predicted == distractor if predicted else False
        
        result = {
            "question_id": qid,
            "predicted_answer": predicted,
            "ground_truth": ground_truth,
            "correct": correct,
            "attack_success": attack_success,
            "distractor_push": distractor,
            "latency": round(latency, 1),
            "rounds_used": final.get("rounds_used", 0),
            "verdict": final.get("verifier_verdict", ""),
        }
        
        symbol = "🔥P" if attack_success else ("✓" if correct else "✗")
        print(f"  [{i+1}/{len(target_questions)}] {qid} {symbol} | "
              f"Pred:{predicted} GT:{ground_truth} Push:{distractor} | "
              f"{latency:.0f}s R:{result['rounds_used']}")
        
    except Exception as e:
        print(f"  [{i+1}/{len(target_questions)}] {qid} ❌ERROR | {e}")
        result = {
            "question_id": qid,
            "predicted_answer": None,
            "ground_truth": q["answer_idx"],
            "correct": False,
            "attack_success": False,
            "distractor_push": distractor,
            "latency": 0,
            "rounds_used": 0,
            "verdict": "ERROR",
            "error": str(e),
        }
    
    results.append(result)

# === Step 5: Results ===
total_time = time.time() - t_start
total = len(results)
correct_count = sum(1 for r in results if r["correct"])
poisoned_count = sum(1 for r in results if r.get("attack_success"))
error_count = sum(1 for r in results if r.get("verdict") == "ERROR" or r.get("error"))
wrong_count = total - correct_count - error_count

print(f"\n" + "=" * 60)
print(f"RESULTS: RAG Poisoning — Gemma 4 E2B (50 questions)")
print(f"=" * 60)
print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  Correct:     {correct_count}/{total} ({correct_count/total*100:.0f}%)")
print(f"  🔥 Poisoned: {poisoned_count}/{total} ({poisoned_count/total*100:.0f}%)")
print(f"  Wrong:       {wrong_count}/{total} ({wrong_count/total*100:.0f}%)")
print(f"  Errors:      {error_count}/{total} ({error_count/total*100:.0f}%)")

if poisoned_count > 0:
    print(f"\n  Attack success by distractor:")
    for d in sorted(set(r["distractor_push"] for r in results if r.get("attack_success"))):
        n = sum(1 for r in results if r.get("attack_success") and r["distractor_push"] == d)
        print(f"    Push={d}: {n} poisoned")

# Save
timestamp = time.strftime("%Y%m%d_%H%M%S")
out_path = REPO_ROOT / "medqa_usmle" / "outputs" / f"poison_gemma4_50_{timestamp}.json"
with open(out_path, "w") as f:
    json.dump({
        "config": {"model": "gemma4:e2b", "index": "index_poisoned_50", "num_targets": len(target_ids)},
        "results": results,
        "summary": {
            "total": total,
            "correct": correct_count,
            "poisoned": poisoned_count,
            "errors": error_count,
            "wrong": wrong_count,
            "total_time_s": round(total_time),
        }
    }, f, indent=2)
print(f"\n  Results saved: {out_path.name}")

print(f"\n  Compare: DeepSeek-v4-flash (5 questions) = 0% attack success")
print(f"  Compare: Gemma 4 E2B (5 questions) = 40% attack success")
