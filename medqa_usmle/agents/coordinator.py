"""
Coordinator Agent — routes the pipeline based on variant and round.

V2 (no memory): rag → reasoning → verifier
V3 (full): memory → rag → reasoning → verifier
V4 (no verifier): memory → rag → reasoning → END
"""

import time

from medqa_usmle.agents.state import AgentState


def coordinator_node(state: AgentState, config: dict) -> dict:
    """
    Coordinator node — routes based on pipeline stage and variant.
    After initial pipeline (memory→rag→reasoning→verifier), routes
    through coordinator so that rounds_used is tracked properly
    and max_rounds limit is enforced (fixes infinite loop bug).
    """
    rounds = state.get("rounds_used", 0)
    max_rounds = state.get("max_rounds", 5)
    config_key = config.get("_variant", "v3")
    has_verifier = config_key in ("v2", "v3")

    if rounds >= max_rounds:
        return {
            "next_agent": "end",
            "degraded": True,
            "error": "Max rounds exceeded",
            "rounds_used": rounds + 1,
        }

    # Define pipeline order per variant
    pipelines = {
        "v2": ["rag", "reasoning", "verifier"],
        "v3": ["memory", "rag", "reasoning", "verifier"],
        "v4": ["memory", "rag", "reasoning"],
    }

    pipeline = pipelines.get(config_key, pipelines["v3"])

    if rounds < len(pipeline):
        if has_verifier and state.get("verifier_verdict"):
            # Verifier has run — use verdict to route
            if state["verifier_verdict"] == "APPROVE":
                next_agent = "end"
            else:
                next_agent = "reasoning"
        else:
            # First time through the pipeline
            next_agent = pipeline[rounds]
    elif has_verifier:
        # Retry loop: verifier → coordinator → reasoning → verifier → ...
        if state.get("verifier_verdict") == "APPROVE":
            next_agent = "end"
        else:
            next_agent = "reasoning"
    else:
        # Past initial pipeline (V4 only, no verifier)
        next_agent = "end"

    # Record trace so the UI graph can show the coordinator as executed
    node_trace = {
        "node": "coordinator",
        "timestamp": time.time(),
        "latency": 0.001,
        "messages_sent": (
            f"Variant={config_key}, round={rounds}, "
            f"verdict={state.get('verifier_verdict', 'none')}"
        ),
        "raw_response": f"route -> {next_agent}",
        "parsed": {
            "next_agent": next_agent,
            "rounds_used": rounds + 1,
        },
    }
    traces = state.get("node_traces", [])
    traces.append(node_trace)

    return {
        "next_agent": next_agent,
        "rounds_used": rounds + 1,
        "node_traces": traces,
    }
