"""
Quick benchmark runner — run all 5 variants on N questions and compare.
"""

import sys, time
from medqa_usmle.variants.v0_direct import run_v0
from medqa_usmle.variants.v1_rag_only import run_v1
from medqa_usmle.variants.v3_full_system import run_v3

N = 3
results = {}

variants = [
    ("v0", lambda: run_v0(split="dev", limit=N)),
    ("v1", lambda: run_v1(split="dev", limit=N)),
    ("v2", lambda: run_v3(split="dev", limit=N, with_memory=False, with_verifier=True)),
    ("v3", lambda: run_v3(split="dev", limit=N, with_memory=True, with_verifier=True)),
    ("v4", lambda: run_v3(split="dev", limit=N, with_memory=True, with_verifier=False)),
]

for name, fn in variants:
    print(f"\n--- {name} ---")
    t0 = time.time()
    try:
        r = fn()
        results[name] = r
    except Exception as e:
        print(f"  ERROR: {e}")
    print(f"  Time: {time.time()-t0:.1f}s")

print("\n" + "="*55)
print("  COMPARISON TABLE")
print("="*55)
print(f"  {'Var':>4} | {'Acc':>6} | {'Lat':>7} | {'Tok':>6} | {'Rnd':>4}")
print(f"  {'----':>4} | {'------':>6} | {'-------':>7} | {'------':>6} | {'----':>4}")
for v in ["v0", "v1", "v2", "v3", "v4"]:
    if v not in results:
        continue
    r = results[v]
    n = len(r)
    acc = sum(1 for x in r if x["correct"]) / n if n else 0
    lat = sum(x.get("latency", 0) for x in r) / n if n else 0
    tok = sum(x.get("output_tokens", 0) for x in r) / n if n else 0
    rnd = sum(x.get("rounds_used", 0) for x in r) / n if n else 0
    print(f"  {v:>4} | {acc:>6.3f} | {lat:>6.1f}s | {tok:>5.0f} | {rnd:>3.0f}")

# Generate report
print("\n" + "="*55)
print("  Generating full report...")
from medqa_usmle.eval.tables import generate_full_report, load_all_variants
from pathlib import Path
output_dir = Path("medqa_usmle") / "outputs"
report = generate_full_report(results, str(output_dir / "benchmark_report.md"))
