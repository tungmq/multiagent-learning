"""
Verifier Agent — gate node that checks candidate answer correctness.

Uses chain-of-thought before verdict and outputs a confidence score.
Only APPROVEs when confidence >= threshold (default 0.7).
Saves full node_trace for later re-evaluation.
"""

import json
import time

from medqa_usmle.agents.state import AgentState
from medqa_usmle.agents.prompts import VERIFIER_SYSTEM_PROMPT
from medqa_usmle.llm import create_provider


def verifier_node(state: AgentState, config: dict) -> dict:
    """Gate node — checks candidate answer before final output."""
    provider = create_provider(config, model_config="verifier_model")

    question = state.get("question_stem", "")
    options = state.get("options", {})
    candidate = state.get("candidate_answer", "")
    explanation = state.get("candidate_explanation", "")
    rag = state.get("rag_context", "")

    # Build verifier prompt with explicit step-by-step instructions
    prompt_parts = [
        f"## Question\n{question}",
    ]
    for letter in ["A", "B", "C", "D"]:
        if letter in options:
            prompt_parts.append(f"{letter}. {options[letter]}")
    prompt_parts.extend([
        f"\n## Candidate Answer\n{candidate}",
        f"\n## Reasoning Provided\n{explanation}",
    ])
    if rag and "placeholder" not in rag and "not found" not in rag:
        prompt_parts.append(f"\n## Medical Knowledge (RAG)\n{rag}")
    prompt_parts.append(
        "\n## Instructions\n"
        "Step 1: Re-read the question and identify the key clinical findings.\n"
        "Step 2: Determine what the question is asking (diagnosis, treatment, mechanism, etc.)\n"
        "Step 3: Evaluate the candidate answer against the options.\n"
        "Step 4: Check if the candidate explanation is medically sound.\n"
        "Step 5: Check if the answer contradicts any RAG medical knowledge.\n"
        "Step 6: Assign a confidence score 0.0-1.0 (how sure you are the answer is correct).\n"
        "Step 7: Output your final verdict.\n"
        "Your ENTIRE response must be ONLY the verdict JSON object — no other text, no code fences.\n"
    )

    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(prompt_parts)},
    ]

    start_t = time.time()
    vm_config = config.get("verifier_model", config["model"])
    resp = provider.invoke(messages, num_predict=vm_config.get("num_predict", 2048))
    latency = time.time() - start_t
    content = resp.content.strip()
    thinking = resp.thinking

    # Parse verdict
    verdict = "RETRY"
    reason = ""
    confidence = 0.0

    if content:
        # Try JSON extraction
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                v = parsed.get("verdict", "").upper()
                if v in ("APPROVE", "RETRY"):
                    verdict = v
                reason = parsed.get("reason", "")
                confidence = float(parsed.get("confidence", 0.0))
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: scan for APPROVE/RETRY outside JSON
        if verdict == "RETRY":
            upper = content.upper()
            if "VERDICT: APPROVE" in upper or "VERDICT:APPROVE" in upper:
                verdict = "APPROVE"
            elif "VERDICT: RETRY" in upper or "VERDICT:RETRY" in upper:
                verdict = "RETRY"

        # Override: if confidence < threshold, force RETRY
        threshold = config.get("verifier", {}).get("confidence_threshold", 0.7)
        if verdict == "APPROVE" and confidence < threshold:
            verdict = "RETRY"
            reason = f"Confidence {confidence:.2f} below threshold {threshold:.2f}"
        elif verdict == "APPROVE" and confidence == 0.0:
            # No confidence in output — require it for APPROVE
            verdict = "RETRY"
            reason = "No confidence score provided"

    # Build node trace
    node_trace = {
        "node": "verifier",
        "timestamp": time.time(),
        "latency": round(latency, 3),
        "prompt_tokens": resp.usage.get("input_tokens", 0),
        "completion_tokens": resp.usage.get("output_tokens", 0),
        "messages_sent": messages,
        "raw_response": content,
        "thinking": thinking,
        "parsed": {
            "verdict": verdict,
            "reason": reason,
            "confidence": confidence,
        },
    }

    traces = state.get("node_traces", [])
    traces.append(node_trace)

    return {
        "verifier_verdict": verdict,
        "verifier_reason": reason or content[:300],
        "verifier_confidence": confidence,
        "node_traces": traces,
        "total_latency": state.get("total_latency", 0) + latency,
        "total_input_tokens": state.get("total_input_tokens", 0) + resp.usage.get("input_tokens", 0),
        "total_output_tokens": state.get("total_output_tokens", 0) + resp.usage.get("output_tokens", 0),
    }
