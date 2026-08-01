# ai-in-sec — Medical Multi-Agent LLM System for MedQA-USMLE

> Research project: Multi-agent LLM architectures for factual accuracy on medical QA.
> Benchmark: **MedQA-USMLE** (1,273 USMLE Step 2CK/3 multiple-choice questions, 4 options).
> Author: Alex (Mai Quế Tùng) — 250202026, UIT

## 🏆 Final Leaderboard (Full Test Set)

| Rank | Variant | Pipeline | Accuracy | Δ vs V0 | Notes |
|:---:|:-------:|----------|:--------:|:-------:|-------|
| 🥇 | **V1** | Single LLM + RAG | **93.09%** | +0.32% | Best overall, simplest |
| 🥈 | **V4** | Memory→RAG→Reasoning | **93.01%** | +0.24% | No verifier, 3.4× faster than V3 |
| 3 | **V0** | Direct LLM (no tools) | **92.77%** | — | Baseline |
| 4 | **V2** | RAG→Reasoning→Verifier | **92.54%** | −0.23% | No memory |
| 5 | **V3** | Memory→RAG→Reasoning→Verifier | **92.07%** | −0.70% | Verifier broken (0 RETRY) |

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | deepseek-v4-flash via opencode-go provider |
| Temperature | 0.0 |
| num_predict | 8192 |
| max_rounds | 5 (V2/V3/V4) |
| Embedding | all-MiniLM-L6-v2, 384-dim, CUDA |
| RAG index | 76,848 chunks from 18 medical textbooks (~110MB) |
| Verifier model | deepseek-v4-flash (same as main — broken by design) |
| Evaluation | 6 parallel workers, 120s/question timeout |

## 📊 Key Findings

1. **Single LLM + RAG (V1) is optimal at 93.09%** — simplest pipeline achieves the highest accuracy.
2. **Verifier is broken AND harmful**: Using the same model for reasoning and verification results in 0 RETRY across all runs. Removing it (V4) improves accuracy by +0.94% and reduces latency 3.4×.
3. **Memory contributes +0.47%** (V4 vs V2: 93.01% vs 92.54%).
4. **deepseek-v4-flash has strong internal medical knowledge** — RAG only adds +0.32% over direct LLM.
5. **num_predict=8192** gives +3.15% over legacy 2048-token config.

## 🧪 Variants

| Variant | File | Description |
|---------|------|-------------|
| V0 | `medqa_usmle/variants/v0_direct.py` | Single LLM call, no tools, no RAG |
| V1 | `medqa_usmle/variants/v1_rag_only.py` | Single LLM + RAG context |
| V2 | `run_v2_parallel.py` | LangGraph: RAG→Reasoning→Verifier (no memory) |
| V3 | `medqa_usmle/variants/v3_full_system.py` | LangGraph: Memory→RAG→Reasoning→Verifier |
| V4 | `run_v4_parallel.py` | LangGraph: Memory→RAG→Reasoning (no verifier) |

## 📁 Project Structure

```
ai-in-sec/
├── medqa_usmle/           # Main evaluation package
│   ├── agents/            # LangGraph agent nodes (coordinator, reasoning, verifier, memory, rag)
│   ├── configs/           # YAML configuration files
│   ├── data/              # Dataset loader
│   ├── eval/              # Evaluation utilities & tables
│   ├── rag/               # RAG retriever (NumpyVectorStore + SentenceTransformer)
│   ├── variants/          # Variant implementations (v0_direct.py, v1_rag_only.py, v3_full_system.py)
│   ├── outputs/           # Prediction JSONL files & run logs
│   └── LEADERBOARD.md     # Detailed leaderboard
├── rag_research/          # RAG index (chunks, embeddings, textbooks)
├── distillation-model-research/  # E4B/E2B finetuning experiments
├── scripts/               # Utility scripts
├── run_v2_parallel.py     # V2 parallel runner
├── run_v3_parallel.py     # V3 parallel runner
├── run_v4_parallel.py     # V4 parallel runner
└── PLAN.md                # Full project plan
```

## 📄 Output Files

| File | Variant | Accuracy |
|------|:-------:|:--------:|
| `outputs/predictions_v1_test_20260705_170752.jsonl` | V1 | 93.09% |
| `outputs/predictions_v4_test_20260709_054550.jsonl` | V4 | 93.01% |
| `outputs/predictions_v0_test_20260706_081729.jsonl` | V0 | 92.77% |
| `outputs/predictions_v2_test_20260709_171530.jsonl` | V2 | 92.54% |
| `outputs/predictions_v3_test_20260708_232248.jsonl` | V3 | 92.07% |

## 🔐 Security Research Context

This system serves as the **midterm project** evaluating multi-agent LLM robustness. Future work:
- Backdoor attacks on the multi-agent pipeline
- RAG poisoning against strong LLM internal knowledge
- Verifier replacement with a separate, lighter model

## 🖥️ Web UI (Docker Compose)

A web-based interactive UI for exploring leaderboards, running single/multiple questions through any variant, and conducting attack experiments.

```
docker compose up -d --build
```

- **Frontend:** http://localhost:80 — React SPA with Dashboard, Pipeline Lab, Attack Lab
- **API:** http://localhost:8000 — FastAPI backend (/docs for Swagger UI)
- **Backend** imports `medqa_usmle/` directly — no modifications to existing code

For the first build only (installs npm + pip deps), subsequent starts are instant.

### Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/dashboard` | Leaderboard table, stratified chart, pairwise comparison, error analysis |
| Pipeline Lab | `/pipeline` | Interactive question runs with variant/config selection, node traces, history |
| Attack Lab | `/attack` | RAG poisoning / prompt injection / LTM corruption experiments with comparison |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./rag_research/index` | `/app/rag_research/index` | RAG index (read-only) |
| `./medqa_usmle/outputs` | `/app/medqa_usmle/outputs` | Prediction JSONL files (read-only) |
| `./dataset` | `/app/dataset` | MedQA dataset files (read-only) |
| `backend_data` (named) | `/app/backend/data` | SQLite run history DB |
| `huggingface_cache` (named) | `/root/.cache/huggingface` | Embedding model cache |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/status` | System status (RAG index, model provider) |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/config` | Current config (keys masked) |
| `GET` | `/api/leaderboard` | Accuracy, latency, tokens per variant |
| `GET` | `/api/leaderboard/stratified` | Step1 vs Step2&3 breakdown |
| `GET` | `/api/leaderboard/errors` | Error pattern analysis |
| `GET` | `/api/leaderboard/pairwise` | Win/Loss/Tie between variants |
| `GET` | `/api/questions` | Paginated MedQA question list |
| `POST` | `/api/run` | Run pipeline on selected questions |
| `GET` | `/api/run/history` | Pipeline run history |
| `POST` | `/api/attack/run` | Run attack experiment |
| `GET` | `/api/attack/types` | Available attack types |
| `GET` | `/api/attack/history` | Attack experiment history |

## 📜 License & Citation

Internal research project — University of Information Technology (UIT), Ho Chi Minh City.
# multiagent-learning
