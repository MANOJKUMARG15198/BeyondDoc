"""OpenAI-compatible chat client for JSON-schema output, used for the Kimi K3 Ragas judge on Fireworks AI.

Fireworks serves `accounts/fireworks/models/kimi-k3` at https://api.fireworks.ai/inference/v1 (serverless). The key
comes from FIREWORKS_API_KEY and is never logged. Fireworks notes that a json_schema response_format disables reasoning
output and that the prompt should also ask for JSON, so the system prompt says so explicitly.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from medical_rag.generation.claude import LLMError

JSON_HINT = "Reply with a single JSON object that matches the requested schema, and nothing else."


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace local "$ref": "#/$defs/X" with the definition (Ragas models are not recursive)."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return walk(copy.deepcopy(defs[ref.split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


class OpenAICompatLLM:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        timeout_s: float = 300.0,
        max_tokens: int = 16000,
        key_name: str = "FIREWORKS_API_KEY",
    ) -> None:
        from openai import OpenAI

        if not api_key:
            raise LLMError(f"{key_name} is not set (add it to your local .env)")
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=3)
        self.model, self.reasoning_effort, self.temperature = model, reasoning_effort, temperature
        self.max_tokens = max_tokens
        self.usage = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}

    @classmethod
    def fireworks_judge(cls, settings: Any) -> OpenAICompatLLM:
        key = settings.fireworks_api_key.get_secret_value() if settings.fireworks_api_key else ""
        return cls(
            api_key=key,
            base_url=settings.fireworks_base_url,
            model=settings.judge_model,
            reasoning_effort=settings.judge_reasoning_effort,
            temperature=settings.judge_temperature,
            timeout_s=settings.llm_timeout_s * 2.5,
        )

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": f"{system}\n{JSON_HINT}"}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": inline_refs(schema)}},
            max_tokens=self.max_tokens,
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage
        cached = 0
        if usage:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
            self.usage["input_tokens"] += usage.prompt_tokens or 0
            self.usage["cached_input_tokens"] += cached
            self.usage["output_tokens"] += usage.completion_tokens or 0
        self.usage["calls"] += 1
        meta = {
            "stop_reason": choice.finish_reason,
            "served_by": response.model,
            "input_tokens": usage.prompt_tokens if usage else None,
            "cached_input_tokens": cached,
            "output_tokens": usage.completion_tokens if usage else None,
        }
        if choice.finish_reason == "length":
            raise LLMError("judge output truncated at max tokens")
        try:
            return json.loads(choice.message.content or ""), meta
        except json.JSONDecodeError as exc:
            raise LLMError("judge did not return valid JSON") from exc


# Fireworks serverless list prices for the judge (USD per 1M tokens); update when the price list changes.
KIMI_K3_PRICE = {"input": 3.00, "cached_input": 0.30, "output": 15.00}


def cost_usd(usage: dict[str, int], price: dict[str, float] = KIMI_K3_PRICE) -> float:
    uncached = max(usage.get("input_tokens", 0) - usage.get("cached_input_tokens", 0), 0)
    total = (
        uncached * price["input"]
        + usage.get("cached_input_tokens", 0) * price["cached_input"]
        + usage.get("output_tokens", 0) * price["output"]
    )
    return round(total / 1_000_000, 4)
