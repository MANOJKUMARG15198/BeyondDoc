"""Ragas adapters: the judge LLM goes through the project's Claude structured-output client (no forced tool use,
no sampling parameters, which current Claude models reject), and embeddings use the pinned bge model."""

from __future__ import annotations

import asyncio
import copy
import hashlib
from contextvars import ContextVar
from typing import Any, TypeVar

from pydantic import BaseModel
from ragas.embeddings import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM

T = TypeVar("T", bound=BaseModel)

JUDGE_SYSTEM = "You are a careful evaluation assistant. Follow the instructions exactly and reply in the given format."
# Set by the caller around each metric call ("q001:run0:faithfulness") so recorded judge outputs can be traced back.
# asyncio tasks and asyncio.to_thread copy the current context, so the tag reaches the worker thread.
JUDGE_TAG: ContextVar[str | None] = ContextVar("judge_tag", default=None)


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Structured outputs need additionalProperties: false and every property required, on every object."""
    schema = copy.deepcopy(schema)

    def walk(node: Any) -> None:  # node is a schema; "properties" and "$defs" map names to schemas
        if isinstance(node, list):
            for value in node:
                walk(value)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        node.pop("title", None)
        node.pop("default", None)
        for key, value in node.items():
            if key in ("properties", "$defs"):
                for sub in value.values():
                    walk(sub)
            elif isinstance(value, dict | list):
                walk(value)

    walk(schema)
    return schema


class ClaudeJudge(InstructorBaseRagasLLM):
    """Ragas judge over any LLM with complete_json (Claude or the Kimi K3 Fireworks client)."""

    def __init__(self, llm: Any, max_concurrency: int | None = None, record: bool = False) -> None:
        self.llm = llm
        self.calls = 0
        self.max_concurrency = max_concurrency
        self._sem: asyncio.Semaphore | None = None
        # Opt-in (record=True): keeps every judge output with its tag and a hash of the prompt (not the prompt text).
        self.record = record
        self.trace: list[dict[str, Any]] = []

    def generate(self, prompt: str, response_model: type[T]) -> T:
        self.calls += 1
        output, _ = self.llm.complete_json(JUDGE_SYSTEM, prompt, strict_schema(response_model.model_json_schema()))
        result = response_model.model_validate(output)
        if self.record:
            self.trace.append(
                {
                    "tag": JUDGE_TAG.get(),
                    "schema": response_model.__name__,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
                    "prompt_chars": len(prompt),
                    "output": output,
                }
            )
        return result

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        if not self.max_concurrency:
            return await asyncio.to_thread(self.generate, prompt, response_model)
        if self._sem is None:  # created inside the running event loop
            self._sem = asyncio.Semaphore(self.max_concurrency)
        async with self._sem:
            return await asyncio.to_thread(self.generate, prompt, response_model)


class BgeRagasEmbedding(BaseRagasEmbedding):
    """Symmetric question-to-question similarity: no query instruction."""

    def __init__(self, embedder: Any) -> None:
        super().__init__()
        self.embedder = embedder

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return [float(x) for x in self.embedder.embed_documents([text]).vectors[0]]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)
