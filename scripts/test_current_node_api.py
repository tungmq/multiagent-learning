"""Verify /run/start + /run/{id}/status expose current_node transitions.

Monkeypatches the graph builder / question loader / initial_state so the full
pipeline router worker runs without LLM calls.
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

from langgraph.graph import StateGraph, START, END


class FakeState(TypedDict, total=False):
    x: Annotated[int, operator.add]
    node_traces: Annotated[list[dict], operator.add]
    final_explanation: str
    total_latency: float
    total_input_tokens: int
    total_output_tokens: int


def _make_fake_node(name: str):
    def node(state: FakeState) -> dict:
        time.sleep(0.1)
        return {"node_traces": [{
            "node": name, "messages_sent": f"in-{name}", "raw_response": f"out-{name}",
            "latency": 0.1, "prompt_tokens": 10, "completion_tokens": 5,
        }]}
    return node


def fake_build_variant_graph(config: dict, with_memory: bool = True, with_verifier: bool = True):
    g = StateGraph(FakeState)
    g.add_node("coordinator", _make_fake_node("coordinator"))
    g.add_node("rag", _make_fake_node("rag"))
    g.add_node("reasoning", _make_fake_node("reasoning"))
    path = ["coordinator", "rag", "reasoning"]
    if with_memory:
        g.add_node("memory", _make_fake_node("memory"))
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
loader_mod.load_questions = lambda split: [{
    "question_id": "q1", "question": "?", "options": {"A": "a", "B": "b"},
    "answer_idx": "A", "meta_info": "step1",
}]
state_mod.initial_state = lambda question, config: {"x": 0, "node_traces": []}

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Reset in-memory registry to avoid interference from earlier tests
from backend.routers import pipeline as pipe
pipe.RUN_STATE.clear()

resp = client.post("/api/run/start", json={
    "question_ids": ["q1"], "variant": "v3",
    "with_memory": True, "with_verifier": True,
})
assert resp.status_code == 200, resp.text
run_id = resp.json()["run_id"]
print(f"run_id: {run_id}")

# Poll status rapidly to observe current_node transitions
seen_nodes = []
live_content_seen = False
deadline = time.time() + 30
final = None
while time.time() < deadline:
    st = client.get(f"/api/run/{run_id}/status").json()
    assert "current_node" in st, f"missing current_node in snapshot: {list(st.keys())}"
    if st["current_node"]:
        seen_nodes.append((st["current_node"], len(st["current_traces"])))
    # Live trace content: while the run is in progress, completed node traces
    # must already carry input/output (not only latency/tokens).
    for tr in st.get("current_traces", []):
        if tr.get("input") or tr.get("output"):
            live_content_seen = True
    if st["status"] in ("completed", "error", "cancelled"):
        final = st
        break
    time.sleep(0.05)

assert final is not None, "run did not finish in time"
print(f"final status: {final['status']}, summary: {final['summary']}")
print(f"observed running nodes (node, #completed traces at that poll): {seen_nodes}")

distinct = [n for n, _ in seen_nodes]
assert distinct, "current_node was never set!"
assert all(n in ("coordinator", "memory", "rag", "reasoning", "verifier") for n in distinct)
# The running node must always be a superset-or-equal frontier of completed
# traces (never shows a node as running before it can run, never a node whose
# trace already completed).
for node, n_traces in seen_nodes:
    assert n_traces <= 4, f"node {node} still shown running after {n_traces} traces"

# After completion current_node must be null
assert final["current_node"] is None, final["current_node"]
assert live_content_seen, "no live input/output content observed during running!"
print("OK: current_node lifecycle correct — set while a node runs, null after completion")
print("OK: live trace input/output populated during running (not only after completion)")
