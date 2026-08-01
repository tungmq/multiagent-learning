"""
Reasoning Agent — medical MCQ reasoning.

Takes question + RAG context + LTM context, outputs candidate answer + explanation.
Saves full node_trace (messages, response, thinking) for later re-evaluation.
"""

import json
import time

from medqa_usmle.agents.state import AgentState
from medqa_usmle.agents.prompts import REASONING_SYSTEM_PROMPT, build_answer_prompt
from medqa_usmle.llm import create_provider


def reasoning_node(state: AgentState, config: dict) -> dict:
    """Medical reasoning node — generates candidate answer."""
    provider = create_provider(config, model_config="reasoning_model")

    rag = state.get("rag_context", "") or ""
    ltm = state.get("ltm_context", "") or ""
    verifier_feedback = state.get("verifier_reason", "") or ""

    is_retry = state.get("verifier_verdict") == "RETRY"
    if is_retry and "rounds_used" in state:
        feedback = f"Previous answer was rejected. {verifier_feedback}"
        prompt = build_answer_prompt(
            state["question_stem"],
            state["options"],
            rag_context=rag,
            ltm_context=ltm,
            verifier_feedback=feedback,
        )
    else:
        feedback = ""
        prompt = build_answer_prompt(
            state["question_stem"],
            state["options"],
            rag_context=rag,
            ltm_context=ltm,
        )

    messages = [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    start_t = time.time()
    rm_config = config.get("reasoning_model", config["model"])
    resp = provider.invoke(messages, num_predict=rm_config.get("num_predict", 4096))
    latency = time.time() - start_t
    content = resp.content.strip()
    thinking = resp.thinking

    # Extract answer from JSON or raw text
    answer = None
    explanation = ""

    if content:
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                answer = parsed.get("answer", "")
                explanation = parsed.get("explanation", "") or parsed.get("reason", "")
        except (json.JSONDecodeError, ValueError):
            pass

        if not answer:
            for letter in ["A", "B", "C", "D"]:
                if letter in content.upper():
                    answer = letter
                    break

        if not explanation:
            explanation = content[:300]

    # Extract explanation from thinking if still empty
    if not explanation and thinking:
        # Last substantive line from thinking
        lines = [l.strip() for l in thinking.split("\n") if l.strip()]
        explanation = lines[-1][:300] if lines else ""

    # Build node trace
    node_trace = {
        "node": "reasoning",
        "timestamp": time.time(),
        "latency": round(latency, 3),
        "prompt_tokens": resp.usage.get("input_tokens", 0),
        "completion_tokens": resp.usage.get("output_tokens", 0),
        "is_retry": is_retry,
        "verifier_feedback": feedback if is_retry else "",
        "messages_sent": messages,
        "raw_response": content,
        "thinking": thinking,
        "parsed": {
            "answer": answer or "",
            "explanation": explanation,
        },
    }

    traces = state.get("node_traces", [])
    traces.append(node_trace)

    return {
        "candidate_answer": answer or "",
        "candidate_letter": answer if answer and answer in "ABCD" else "",
        "candidate_explanation": explanation,
        "node_traces": traces,
        "total_latency": state.get("total_latency", 0) + latency,
        "total_input_tokens": state.get("total_input_tokens", 0) + resp.usage.get("input_tokens", 0),
        "total_output_tokens": state.get("total_output_tokens", 0) + resp.usage.get("output_tokens", 0),
    }
