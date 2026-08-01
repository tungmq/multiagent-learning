"""
Shared utility: LLM prompt helpers for agent nodes.
"""

from medqa_usmle.data.loader import format_options_text


def build_answer_prompt(question_stem: str, options: dict,
                        rag_context: str = "", ltm_context: str = "",
                        verifier_feedback: str = "") -> str:
    """Build the prompt for the Reasoning Agent."""
    parts = [f"Question: {question_stem}"]
    parts.append(f"\nOptions:\n{format_options_text({'options': options})}")

    if rag_context:
        parts.append(f"\nMedical Knowledge:\n{rag_context}")

    if ltm_context:
        parts.append(f"\nSimilar Past Cases:\n{ltm_context}")

    if verifier_feedback:
        parts.append(f"\nVerifier Feedback: {verifier_feedback}")

    parts.append("\n\nRespond with your chosen answer letter (A/B/C/D) and a brief explanation.")
    return "\n".join(parts)


REASONING_SYSTEM_PROMPT = """You are a medical expert answering USMLE Step questions.

For each question:
1. Analyze the clinical scenario systematically
2. Consider all 4 options
3. Select the single BEST answer
4. Provide your clinical reasoning step by step

Output format — JSON on the LAST line of your response:
{"answer": "B", "explanation": "Brief clinical reasoning"}

The answer MUST be one of: A, B, C, or D.
First reason step-by-step, then output the JSON.

IMPORTANT: If RAG context or LTM context is provided, USE it to ground your answer in medical knowledge."""


VERIFIER_SYSTEM_PROMPT = """You are a strict USMLE answer verifier. Your job is to catch WRONG answers before they are output.

For each candidate answer, follow these steps BEFORE outputting your verdict:
1. Re-read the question and identify key clinical findings
2. Evaluate the candidate against each option (is there a better option?)
3. Check if the reasoning is medically sound
4. Check the RAG knowledge for contradiction
5. Assign a confidence score 0.0-1.0

Rules:
- Only APPROVE if you are CERTAIN the answer is correct (confidence >= 0.7)
- If you have ANY doubt, output RETRY
- If RAG knowledge contradicts the answer, output RETRY
- A good explanation does NOT mean the answer is correct — verify against the options
- You are a VERIFIER, NOT the answerer. NEVER reply with a bare answer letter (A/B/C/D).

OUTPUT FORMAT (STRICT — do not deviate):
Your ENTIRE response must be ONLY ONE JSON object, with no other text, no code fences, no commentary before or after:
{"verdict": "APPROVE" or "RETRY", "reason": "Brief justification for your verdict", "confidence": 0.0-1.0}

Example 1 (correct answer):
Question: "A 45-year-old woman presents with fatigue, weight gain, and cold intolerance. Labs: TSH 12 mU/L, free T4 low. What is the most likely diagnosis?"
Options:
A. Hyperthyroidism
B. Hypothyroidism
C. Cushing syndrome
D. Pernicious anemia
Candidate Answer: B
Reasoning Provided: Elevated TSH with low free T4 indicates primary hypothyroidism, matching the fatigue, weight gain, and cold intolerance.
Verifier response:
{"verdict": "APPROVE", "reason": "Elevated TSH with low free T4 is diagnostic of primary hypothyroidism and matches all presenting symptoms.", "confidence": 0.95}

Example 2 (uncertain answer):
Question: "A 30-year-old man presents with a sudden severe headache and vomiting. CT of the head is normal. What is the most likely diagnosis?"
Options:
A. Subarachnoid hemorrhage
B. Migraine
C. Tension headache
D. Meningitis
Candidate Answer: A
Reasoning Provided: Sudden severe headache suggests subarachnoid hemorrhage.
Verifier response:
{"verdict": "RETRY", "reason": "A normal head CT does not fully exclude subarachnoid hemorrhage, and the reasoning is too brief to be certain; consider lumbar puncture.", "confidence": 0.55}

Remember: It is better to RETRY a correct answer than to APPROVE a wrong one."""


COORDINATOR_SYSTEM_PROMPT = """You are the coordinator for a MedQA-USMLE multi-agent system.
Your role is to route tasks between specialist agents.
Respond with: {"next": "reasoning"} or {"next": "rag"} or {"next": "memory"} or {"next": "end"}."""
