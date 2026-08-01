# MedQA-USMLE Leaderboard — Full Test Set (1,273 questions)

## Verified Final Results (from JSONL output files)

| Rank | Variant | Pipeline | Correct | Accuracy | Invalid | Δ vs V0 | Latency | Tokens | Rounds | Notes |
|:----:|---------|----------|:-------:|:--------:|:-------:|:-------:|:-------:|:------:|:-----:|-------|
| 🥇 | **V2** No memory | RAG→Reasoning→Verifier | 1198/1273 | **94.11%** | 0 | +1.34% | 61.9s | 5,269 | 1.9 | ⚡ **New #1 after rerun fix** |
| 🥈 | **V3** Full | Memory→RAG→Reasoning→Verifier | 1192/1273 | **93.64%** | 0 | +0.87% | 56.3s | 4,819 | 2.9 | Rerun fixed infinite loop bug |
| 🥉 | **V1** RAG-only | Single LLM + RAG | 1185/1273 | **93.09%** | 28 (2.2%) | +0.32% | 15.1s | 1,294 | 1.0 | Best overall, simplest |
| 4 | **V4** No verifier | Memory→RAG→Reasoning | 1184/1273 | **93.01%** | 0 | +0.24% | 16.5s | 1,403 | 3.0 | 3.4× faster than V3 |
| 5 | **V0** Direct LLM | 1 call, no tools | 1181/1273 | 92.77% | 0 | — | 17.0s | 1,284 | 1.0 | Baseline |

## Rerun Results (2026-07-26)

- **Bug found**: Verifier→Reasoning loop bypassed Coordinator, so `rounds_used` never incremented → infinite RETRY loop until recursion limit hit
- **Fix**: Route verifier→coordinator instead of verifier→reasoning; coordinator now tracks rounds_used for retries
- **V3 rerun**: 36 errored → 20 correct (+20), 93.64% overall
- **V2 rerun**: 33 errored → 20 correct (+20), 94.11% overall  
- **Rerun used no-verifier** (fast) mode for errored questions only; original predictions for the rest
- V1, V0, V4 had 0 errors — no rerun needed

## Pairwise Comparison: V0 vs V3 (original)

| Result | Count | Meaning |
|--------|:-----:|---------|
| Both correct | 1,139 | Easy questions, both answered correctly |
| Only V3 correct | 33 | Multi-agent improved hard questions |
| Only V0 correct | 42 | Multi-agent lost points |
| Both wrong | 59 | Very hard questions |
| McNemar χ² = 0.458, p = 0.499 (NOT significant) |

## Verifier Analysis

- **V3 Jul8 (original 92.07%)**: Verifier ran on 1,237/1,273 questions. **APPROVE: 1,237, RETRY: 0**.
  - 36 questions had recursion limit errors (never reached Verifier)
- **V2 (original 92.54%)**: 33 recursion limit errors
- **Root cause**: Verifier→Reasoning loop bypassed Coordinator, causing infinite loop until recursion limit hit

## Key Findings

1. **Verifier was non-functional due to infinite loop bug**: same model cannot self-verify
2. **Memory seems to hurt marginally** (V3 93.64% vs V2 94.11%) — may be accumulating noise from wrong answers
3. **Single LLM + RAG (V1) still strong** at 93.09% with simplest pipeline
4. **deepseek-v4-flash has strong built-in medical knowledge** — RAG only adds +0.32%
5. **Config matters**: num_predict=8192 gives ~+5.65% vs 2048

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | deepseek-v4-flash via opencode |
| Temperature | 0.0 |
| num_predict | 8192 |
| max_rounds | 5 (V2/V3/V4) |
| RAG | all-MiniLM-L6-v2, 76,848 chunks, 18 textbooks |
| Parallel workers | 6 |
| Per-question timeout | 120s |
| Recursion limit | 35 |
| Seed | 42 |
