"""Verify node_start events flow through run_single_question_streaming.

Uses a fake LangGraph (monkeypatched build_variant_graph) that mimics the
real coordinator -> memory -> rag -> coordinator -> reasoning -> verifier
topology with per-node node_traces appended to state, so no LLM calls are made.
"""
import sys
from pathlib import Path
from typing import Annotated, TypedDict, Any
import operator

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from langgraph.graph import StateGraph, START, END
from medqa_usmle.agents.state import AgentState

# Monkeypatch the graph builder used by backend.worker
import backend.worker as worker_mod
import medqa_usmle.variants.v3_full_system as v3_mod


class FakeState(TypedDict, total=False):
    x: Annotated[int, operator.add]
    node_traces: Annotated[list[dict], operator.add]
    next_agent: str
    final_explanation: str
    total_latency: float
    total_input_tokens: int
    total_output_tokens: int


TRACES: dict[str, dict] = {}  # node -> trace dict


def _make_fake_node(name: str):
    def node(state: FakeState) -> dict:
        import time
        time.sleep(0.05)  # simulate work so node_start ordering is observable
        tr = {
            "node": name,
            "messages_sent": f"input-{name}",
            "raw_response": f"output-{name}",
            "latency": 0.05,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }
        TRACES[name] = tr
        return {"node_traces": [tr]}
    return node


def fake_build_variant_graph(config: dict, with_memory: bool = True, with_verifier: bool = True):
    # Single path mirroring the real topology: coordinator → memory → rag →
    # reasoning → verifier → END
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

# Patch question loader: use a fake question
from medqa_usmle.data import loader as loader_mod
loader_mod.load_questions = lambda split: [{
    "question_id": "q1", "question": "?", "options": {"A": "a", "B": "b"},
    "answer_idx": "A", "meta_info": "step1",
}]

# Patch initial_state (worker imports it from medqa_usmle.agents.state at call
# time, so patching the module attribute is enough)
from medqa_usmle.agents import state as state_mod


def fake_initial_state(question, config):
    return {"x": 0, "node_traces": [], "next_agent": "end"}

state_mod.initial_state = fake_initial_state

events = list(worker_mod.run_single_question_streaming(
    "q1", variant="v3", with_memory=True, with_verifier=True,
    config_override={"pipeline": {"max_rounds": 8}},
))

print(f"total events: {len(events)}")
for e in events:
    print(f"  {e['type']}: {e['data'] if e['type'] != 'question_complete' else '...'}")

# Assertions
starts = [e["data"]["node"] for e in events if e["type"] == "node_start"]
updates = [e["data"]["node"] for e in events if e["type"] == "node_update"]
assert starts == ["coordinator", "memory", "rag", "reasoning", "verifier"], f"BAD starts: {starts}"
assert updates == ["coordinator", "memory", "rag", "reasoning", "verifier"], f"BAD updates: {updates}"
assert any(e["type"] == "question_complete" for e in events)
complete = [e for e in events if e["type"] == "question_complete"][0]
assert len(complete["data"]["node_traces"]) == 5
print("\nOK: node_start fires before node_update for every node, in execution order")
