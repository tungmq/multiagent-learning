"""Verify Web-UI LTM memory lifecycle (fix: memory now works across questions).

Scenario: run 2 questions via /api/run/start with v3.
  - Q1 is the first question: memory node must show NO prior context (by design).
  - After Q1 completes correctly it is recorded into LTM.
  - Q2's memory node must then find Q1 as a similar past case (ltm_hits=1).

Monkeypatches the graph builder / question loader / initial_state so no LLM
calls happen; the REAL memory node runs (needs sentence-transformers locally).

Run: /tmp/venv-test/bin/python scripts/test_ltm_memory_ui.py
"""
import sys, time
from pathlib import Path
from typing import Annotated, TypedDict
import operator

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import backend.worker as worker_mod
import medqa_usmle.variants.v3_full_system as v3_mod
from medqa_usmle.data import loader as loader_mod
from medqa_usmle.agents import state as state_mod
from medqa_usmle.agents import memory as memory_mod

from langgraph.graph import StateGraph, START, END


class FakeState(TypedDict, total=False):
    x: Annotated[int, operator.add]
    question_stem: str
    ground_truth: str
    rounds_used: int
    rag_context: str
    predicted_answer: str
    stm_context: str
    ltm_context: str
    raw_ltm_entries: list
    next_agent: str
    node_traces: Annotated[list[dict], operator.add]
    final_explanation: str
    total_latency: float
    total_input_tokens: int
    total_output_tokens: int
    error: str


def _make_fake_node(name: str):
    def node(state: FakeState) -> dict:
        time.sleep(0.05)
        return {"node_traces": [{
            "node": name, "messages_sent": f"in-{name}", "raw_response": f"out-{name}",
            "latency": 0.05, "prompt_tokens": 5, "completion_tokens": 2,
        }]}
    return node


def fake_build_variant_graph(config: dict, with_memory: bool = True, with_verifier: bool = True):
    g = StateGraph(FakeState)
    g.add_node("coordinator", lambda st: {
        "predicted_answer": st.get("ground_truth"),  # always correct in this test
        "node_traces": [{
            "node": "coordinator", "messages_sent": "in-coordinator",
            "raw_response": "out-coordinator", "latency": 0.05,
            "prompt_tokens": 5, "completion_tokens": 2,
        }],
    })
    g.add_node("rag", _make_fake_node("rag"))
    g.add_node("reasoning", _make_fake_node("reasoning"))
    path = ["coordinator", "rag", "reasoning"]
    if with_memory:
        # REAL memory node — exercises embedding + LTM store/query.
        g.add_node("memory", lambda st: memory_mod.memory_node(st, config))
        path.insert(1, "memory")
    if with_verifier:
        g.add_node("verifier", _make_fake_node("verifier"))
        path.append("verifier")
    g.add_edge(START, "coordinator")
    for a, b in zip(path, path[1:]):
        g.add_edge(a, b)
    g.add_edge(path[-1], END)
    return g.compile()


v3_mod.build_variant_graph = fake_build_variant_graph
loader_mod.load_questions = lambda split: [
    {"question_id": "q1", "question": "A 30-year-old man presents with chest pain.", "options": {"A": "a", "B": "b"}, "answer_idx": "A", "meta_info": "step1"},
    {"question_id": "q2", "question": "A 30-year-old man presents with chest pain.", "options": {"A": "a", "B": "b"}, "answer_idx": "B", "meta_info": "step1"},
]
state_mod.initial_state = lambda question, config: {
    "question_stem": question["question"],
    "ground_truth": question["answer_idx"],
    "rounds_used": 0,
    "rag_context": "",
    "node_traces": [],
}
worker_mod._extract_answer_from_state = lambda st: st.get("predicted_answer")

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
from backend.routers import pipeline as pipe
pipe.RUN_STATE.clear()

# LTM must start empty
assert len(memory_mod._LTM_STORE) == 0, "LTM should start empty"

resp = client.post("/api/run/start", json={
    "question_ids": ["q1", "q2"], "variant": "v3",
    "with_memory": True, "with_verifier": True,
})
assert resp.status_code == 200, resp.text
run_id = resp.json()["run_id"]
print(f"run_id: {run_id}")

deadline = time.time() + 60
final = None
while time.time() < deadline:
    st = client.get(f"/api/run/{run_id}/status").json()
    if st["status"] in ("completed", "error", "cancelled"):
        final = st
        break
    time.sleep(0.1)
assert final is not None and final["status"] == "completed", final

# 1) Both questions answered correctly -> both recorded into LTM
assert len(memory_mod._LTM_STORE) == 2, f"LTM store size = {len(memory_mod._LTM_STORE)}, want 2"
print(f"LTM store: {len(memory_mod._LTM_STORE)} entries after run")

# 2) Q1 memory trace: no prior context (by design, first question)
q1_traces = [r for r in final["current_traces"] if r.get("question_id") == "q1"] if "current_traces" in final else []
print(f"Q1 traces found: {len(q1_traces)}")

# Pull per-question traces from the detail endpoint
detail = client.get(f"/api/run/{run_id}").json()
results = detail.get("results", [])
assert len(results) == 2, f"expected 2 results, got {len(results)}"
assert all(r["correct"] for r in results), f"both should be correct: {results}"

mem_traces = {}
for r in results:
    traces = r.get("node_traces", [])
    if isinstance(traces, str):  # stored as JSON string in SQLite
        import json as _json
        traces = _json.loads(traces)
    for tr in traces:
        if isinstance(tr, dict) and tr.get("node") == "memory":
            mem_traces[r["question_id"]] = tr

q1_mem = mem_traces.get("q1")
q2_mem = mem_traces.get("q2")
assert q1_mem is not None and q2_mem is not None, f"memory traces missing: {list(mem_traces)}"

assert "No prior context" in q1_mem["output"], f"Q1 memory should show no prior context: {q1_mem['output']!r}"
print(f"Q1 memory output: {q1_mem['output'][:60]!r}")

# 3) Q2 memory trace: must have found Q1 as a similar past case (LTM context
#    replaces the STM fallback in raw_response when a hit exists)
assert "Similar Past Cases" in q2_mem["output"], f"Q2 memory should show LTM case: {q2_mem['output']!r}"
print(f"Q2 memory output: {q2_mem['output'][:80]!r}")

print("\nOK: Web-UI LTM lifecycle works — Q2 sees Q1 as past memory, UI shows LTM case")
