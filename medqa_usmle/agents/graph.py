"""
LangGraph StateGraph builder for MedQA-USMLE Multi-Agent System.

Uses a closure pattern to inject config into node functions.
"""

from functools import partial
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import RunnableConfig

from medqa_usmle.agents.state import AgentState
from medqa_usmle.agents.coordinator import coordinator_node
from medqa_usmle.agents.reasoning import reasoning_node
from medqa_usmle.agents.verifier import verifier_node
from medqa_usmle.agents.memory import memory_node
from medqa_usmle.agents.rag import rag_node


def _make_node(fn, config: dict):
    """Wrap a node function with config injection."""
    def wrapped(state: AgentState, _config: RunnableConfig = None):
        return fn(state, config)
    return wrapped


def route_after_coordinator(state: AgentState) -> str:
    next_agent = state.get("next_agent", "end")
    # LangGraph conditional edge maps return value to node name or END
    return next_agent


def route_after_verifier(state: AgentState) -> str:
    verdict = state.get("verifier_verdict", "")
    rounds = state.get("rounds_used", 0)
    max_rounds = state.get("max_rounds", 5)

    if verdict == "APPROVE":
        return "end"
    elif rounds >= max_rounds:
        return "end"
    else:
        return "reasoning"


def build_graph(config: dict) -> StateGraph:
    """Build and compile the LangGraph StateGraph with the given config."""
    workflow = StateGraph(AgentState)

    # Add nodes with config injected
    workflow.add_node("coordinator", _make_node(coordinator_node, config))
    workflow.add_node("reasoning", _make_node(reasoning_node, config))
    workflow.add_node("verifier", _make_node(verifier_node, config))
    workflow.add_node("memory", _make_node(memory_node, config))
    workflow.add_node("rag", _make_node(rag_node, config))

    # START → coordinator
    workflow.add_edge(START, "coordinator")

    # Coordinator routing → specialist nodes or END
    workflow.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "memory": "memory",
            "rag": "rag",
            "reasoning": "reasoning",
            "verifier": "verifier",
            "end": END,
        },
    )

    # Memory and RAG return to coordinator
    workflow.add_edge("memory", "coordinator")
    workflow.add_edge("rag", "coordinator")

    # Reasoning → Verifier
    workflow.add_edge("reasoning", "verifier")

    # Verifier gate: APPROVE → END, RETRY → reasoning
    workflow.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "end": END,
            "reasoning": "reasoning",
        },
    )

    return workflow.compile()


def run_graph_single(
    initial_state: dict,
    config: dict,
    recursion_limit: int = 20,
) -> dict:
    graph = build_graph(config)
    result = graph.invoke(
        initial_state,
        {"recursion_limit": recursion_limit},
    )
    return result
