"""Kimi K3 judge on Fireworks: request shape, $ref inlining, cost, errors (fake OpenAI client, no network)."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from medical_rag.evaluation.ragas_adapters import ClaudeJudge, strict_schema
from medical_rag.generation.claude import LLMError
from medical_rag.generation.openai_compat import JSON_HINT, OpenAICompatLLM, cost_usd, inline_refs


class Verdicts(BaseModel):
    class V(BaseModel):
        statement: str
        verdict: int

    statements: list[V]


def fake_llm(content='{"statements": [{"statement": "s", "verdict": 1}]}', finish="stop"):
    llm = OpenAICompatLLM.__new__(OpenAICompatLLM)
    llm.model, llm.reasoning_effort, llm.temperature, llm.max_tokens = (
        "accounts/fireworks/models/kimi-k3",
        "low",
        0.0,
        1000,
    )
    llm.usage = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=20, prompt_tokens_details=SimpleNamespace(cached_tokens=40)
            ),
            model=llm.model,
        )

    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return llm, calls


def test_inline_refs_removes_defs():
    schema = inline_refs(strict_schema(Verdicts.model_json_schema()))
    assert "$defs" not in schema and "$ref" not in str(schema)
    item = schema["properties"]["statements"]["items"]
    assert item["required"] == ["statement", "verdict"] and item["additionalProperties"] is False


def test_request_shape_for_fireworks():
    llm, calls = fake_llm()
    assert ClaudeJudge(llm).generate("prompt", Verdicts).statements[0].verdict == 1
    kw = calls[0]
    assert kw["model"] == "accounts/fireworks/models/kimi-k3"
    assert kw["reasoning_effort"] == "low" and kw["temperature"] == 0.0 and kw["max_tokens"] == 1000
    assert kw["response_format"]["type"] == "json_schema" and "schema" in kw["response_format"]["json_schema"]
    assert kw["messages"][0]["content"].endswith(JSON_HINT)
    assert llm.usage == {"calls": 1, "input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20}


def test_cost_uses_kimi_k3_fireworks_prices():
    # 60 uncached x $3 + 40 cached x $0.30 + 20 output x $15, per 1M tokens
    assert cost_usd({"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20}) == round(
        (60 * 3 + 40 * 0.3 + 20 * 15) / 1e6, 4
    )
    assert cost_usd({"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 1_000_000}) == 18.0


def test_truncated_or_invalid_output_raises():
    with pytest.raises(LLMError):
        fake_llm(finish="length")[0].complete_json("s", "u", {"type": "object"})
    with pytest.raises(LLMError):
        fake_llm(content="not json")[0].complete_json("s", "u", {"type": "object"})


def test_missing_key_is_a_clear_error():
    with pytest.raises(LLMError, match="FIREWORKS_API_KEY"):
        OpenAICompatLLM(api_key="", base_url="https://api.fireworks.ai/inference/v1", model="m")
