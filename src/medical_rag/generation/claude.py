"""Claude API client for JSON-constrained answers (structured outputs, server-side refusal fallback)."""

from __future__ import annotations

import json
from typing import Any

from medical_rag.core.config import Settings


class LLMError(RuntimeError):
    pass


class ClaudeLLM:
    def __init__(self, settings: Settings, *, max_tokens: int = 4000) -> None:
        import anthropic

        if settings.anthropic_api_key is None:
            raise LLMError("ANTHROPIC_API_KEY is not set (add it to your local .env)")
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(), timeout=settings.llm_timeout_s
        )
        self.model = settings.llm_model
        self.effort = settings.llm_effort
        self.max_tokens = max_tokens

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        import anthropic

        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIConnectionError as exc:  # network/DNS failure or timeout, after the client's retries
            raise LLMError("the answer service could not be reached; please try again shortly") from exc
        except anthropic.APIStatusError as exc:  # e.g. 429 rate limit, 5xx/overloaded, after the client's retries
            raise LLMError(f"the answer service returned an error (HTTP {exc.status_code}); please try again") from exc
        meta = {
            "stop_reason": response.stop_reason,
            "served_by": response.model,
            "request_id": response._request_id,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        if response.stop_reason == "refusal":
            return {}, meta
        if response.stop_reason == "max_tokens":
            raise LLMError("answer truncated at max_tokens")
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            return json.loads(text), meta
        except json.JSONDecodeError as exc:
            raise LLMError("model did not return valid JSON") from exc
