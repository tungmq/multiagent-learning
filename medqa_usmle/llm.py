"""LLM Provider Abstraction Layer.

Supports multiple backends:
- Ollama (local, handles `thinking` field)
- OpenAI-compatible (OpenRouter, opencode, etc.)

Design: swap via config.yaml model.provider + model.name
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import requests


# === Retry helper ===
def _request_with_retry(method, url, max_retries=2, base_delay=1.0, **kwargs):
    """HTTP request with exponential backoff on server errors, connection errors, and timeouts."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = method(url, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            last_error = e
            if attempt < max_retries and (
                isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))
                or (isinstance(e, requests.exceptions.HTTPError) and e.response is not None
                    and e.response.status_code >= 500)
            ):
                delay = base_delay * (2 ** attempt)
                print(f"  [Retry] {url} failed (attempt {attempt+1}/{max_retries+1}): {e}. "
                      f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            raise
    raise last_error  # should not reach here


@dataclass
class LLMResponse:
    content: str = ""
    thinking: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)
    latency: float = 0.0


class BaseProvider(ABC):
    @abstractmethod
    def invoke(self, messages: list[dict], **kwargs) -> LLMResponse:
        ...


class OllamaProvider(BaseProvider):
    """Ollama provider — handles Qwen3.x `thinking` field."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.0, keep_alive: str = "30m", num_predict: int = 32):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.num_predict = num_predict

    def invoke(self, messages: list[dict], **kwargs) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("num_predict", self.num_predict),
            },
        }

        start = time.time()
        resp = _request_with_retry(
            requests.post,
            f"{self.base_url}/api/chat",
            max_retries=kwargs.get("max_retries", 2),
            json=payload,
            timeout=kwargs.get("timeout", 120),
        )
        latency = time.time() - start
        data = resp.json()

        msg = data.get("message", {})
        content = msg.get("content", "") or ""
        thinking = msg.get("thinking", "") or ""

        # Qwen3.x puts reasoning in `thinking` and final answer in `content`.
        # If content is empty, fall back to extracting from thinking.
        if not content.strip():
            content = self._extract_answer_from_thinking(thinking)

        return LLMResponse(
            content=content.strip(),
            thinking=thinking.strip(),
            model=data.get("model", self.model),
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
                "total_duration_ns": data.get("total_duration", 0),
            },
            latency=latency,
        )

    @staticmethod
    def _extract_answer_from_thinking(thinking: str) -> str:
        """Extract final answer from Qwen's thinking block."""
        # JSON block with a known control key → pass through as-is (the
        # downstream node (reasoning/verifier/coordinator) parses it).
        import json as _json
        start = thinking.find("{")
        end = thinking.rfind("}") + 1
        if start >= 0 and end > start:
            json_text = thinking[start:end]
            try:
                parsed = _json.loads(json_text)
                if any(k in parsed for k in ("answer", "verdict", "next")):
                    return json_text
            except _json.JSONDecodeError:
                pass
        # Look for common patterns
        lines = thinking.strip().split("\n")
        for line in reversed(lines):
            ul = line.upper().strip()
            # JSON pattern: {"answer": "B"}
            if '"answer"' in ul or "'answer'" in ul:
                for letter in ["A", "B", "C", "D"]:
                    if f'"{letter}"' in ul or f"'{letter}'" in ul:
                        return letter
            # Answer patterns
            if any(f"ANSWER IS {l}" in ul for l in ["A", "B", "C", "D"]):
                for letter in ["A", "B", "C", "D"]:
                    if f"ANSWER IS {letter}" in ul:
                        return letter
            if any(f"ANSWER: {l}" in ul for l in ["A", "B", "C", "D"]):
                for letter in ["A", "B", "C", "D"]:
                    if f"ANSWER: {letter}" in ul:
                        return letter
            # Bare letter on its own line
            stripped = line.strip().upper()
            if stripped in ["A", "B", "C", "D"] and len(stripped) == 1:
                return stripped
            # "B." or "B)" style
            if len(stripped) >= 2 and stripped[0] in "ABCD" and stripped[1] in ".)":
                return stripped[0]
        # Last resort: scan backwards for any letter
        for letter in reversed(["A", "B", "C", "D"]):
            if letter in thinking.upper():
                return letter
        return ""


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI-compatible API provider (OpenRouter, opencode, etc.)."""

    def __init__(self, model: str, base_url: str, api_key: str,
                 temperature: float = 0.0, max_tokens: int = 32):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, messages: list[dict], **kwargs) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Support num_predict as alias for max_tokens (used by reasoning/verifier nodes)
        max_tokens = kwargs.get("max_tokens") or kwargs.get("num_predict") or self.max_tokens
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": max_tokens,
        }

        start = time.time()
        resp = _request_with_retry(
            requests.post,
            f"{self.base_url}/chat/completions",
            max_retries=kwargs.get("max_retries", 2),
            headers=headers,
            json=payload,
            timeout=kwargs.get("timeout", 300),
        )
        latency = time.time() - start
        data = resp.json()

        choice = data["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""

        # deepseek-v4-flash and similar models put final answer in
        # reasoning_content when thinking is enabled. Fall back when
        # content is empty.
        if not content.strip() and reasoning.strip():
            content = self._extract_from_reasoning(reasoning)

        return LLMResponse(
            content=content.strip(),
            thinking=reasoning.strip(),
            model=data.get("model", self.model),
            usage={
                "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
            },
            latency=latency,
        )

    @staticmethod
    def _extract_from_reasoning(reasoning: str) -> str:
        """Extract answer from reasoning_content when content is empty.

        First tries to find JSON block, then falls back to letter extraction.
        """
        # Try to find JSON block first (preserves explanation/verdict)
        import json as _json
        start = reasoning.find("{")
        end = reasoning.rfind("}") + 1
        if start >= 0 and end > start:
            json_text = reasoning[start:end]
            try:
                parsed = _json.loads(json_text)
                if any(k in parsed for k in ("answer", "verdict", "next")):
                    return json_text
            except _json.JSONDecodeError:
                pass

        # Fallback: find letter
        lines = reasoning.strip().split("\n")
        for line in reversed(lines):
            ul = line.upper().strip()
            if '"answer"' in ul:
                for letter in ["A", "B", "C", "D"]:
                    if f'"{letter}"' in ul:
                        return letter
            for pat in [f"ANSWER IS {l}" for l in "ABCD"]:
                if pat in ul:
                    return pat[-1]
            for pat in [f"ANSWER: {l}" for l in "ABCD"]:
                if pat in ul:
                    return pat[-1]
            stripped = line.strip().upper()
            if stripped in ["A", "B", "C", "D"] and len(stripped) == 1:
                return stripped
            if len(stripped) >= 2 and stripped[0] in "ABCD" and stripped[1] in ".).)]":
                return stripped[0]
        for letter in reversed(["A", "B", "C", "D"]):
            if letter in reasoning.upper():
                return letter
        return reasoning[:200]


def create_provider(config: dict, model_config: str = "model") -> BaseProvider:
    """Factory: create provider from config.
    
    Config shape:
      model:
        provider: "ollama" | "opencode" | "openai" | "openrouter"
        name: "qwen3.5:9b" | "deepseek-v4-flash"
        base_url: ... (optional, for OpenAI-compatible)
        api_key: ... (optional, for OpenAI-compatible)
        temperature: 0.0
        num_predict: 32  # max output tokens
    
    Args:
        config: Full configuration dict
        model_config: Key in config dict, default "model".
                      Pass "verifier_model" to use a separate model config.
                      If the key is absent, falls back to "model" — this lets
                      per-agent keys (e.g. "reasoning_model") be optional.
    """
    mc = config.get(model_config) or config["model"]
    provider_type = mc.get("provider", "ollama")
    model_name = mc["name"]

    if provider_type == "ollama":
        return OllamaProvider(
            model=model_name,
            temperature=mc.get("temperature", 0.0),
            keep_alive=mc.get("keep_alive", "30m"),
            num_predict=mc.get("num_predict", 32),
        )
    elif provider_type in ("opencode", "openai", "openrouter"):
        base_url = mc.get("base_url", "https://api.openai.com/v1")
        api_key = mc.get("api_key", "")
        return OpenAICompatibleProvider(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=mc.get("temperature", 0.0),
            max_tokens=mc.get("num_predict", 32),
        )
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
