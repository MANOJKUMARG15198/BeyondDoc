"""What the RAG stages put into a trace (metadata only; text goes through TraceHandle.text, which is None unless
MEDRAG_TRACE_CONTENT=true). Kept apart from the call sites so /search, /ask and the chat turn record the same
fields."""

from __future__ import annotations

from typing import Any

from medical_rag.observability.tracing import TraceHandle, usage_details


def tags(settings: Any) -> list[str]:
    return [f"env:{settings.env}", f"prompt:{settings.answer_prompt}", f"plan:{settings.retrieval_plan}"]


def retrieval_metadata(result: Any, plan: Any, *, mode: str, k: int) -> dict[str, Any]:
    hits = result.hits
    return {
        "mode": mode,
        "k": k,
        "plan": list(getattr(plan, "reasons", ()) or ()),
        "timings_ms": dict(result.timings_ms),
        "chunk_ids": [h.chunk_id for h in hits],
        "document_ids": [h.source.get("document_id") for h in hits],
        "sections": [str(h.source.get("section_path") or "")[:80] for h in hits],
        "selection": dict(result.selection or {}),
    }


def retrieval_output(handle: TraceHandle, result: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"passages": len(result.hits)}
    if handle.content:  # content capture on: include short passage text
        out["passages_text"] = [handle.text(h.source.get("text", "")[:500]) for h in result.hits]
    return out


def answer_metadata(answer: Any) -> dict[str, Any]:
    meta = answer.llm_meta or {}
    return {
        "prompt_version": meta.get("prompt_version"),
        "coverage": meta.get("coverage"),
        "abstained": answer.abstained,
        "abstain_reason": answer.abstain_reason,
        "cited": [c.source_id for c in answer.citations],
        "invalid_citations": meta.get("invalid_citations") or [],
        "served_by": meta.get("served_by"),
        "stop_reason": meta.get("stop_reason"),
        "answer_chars": len(answer.answer or ""),
        "llm_ms": (answer.timings_ms or {}).get("llm_ms"),
    }


def record_answer(span: Any, handle: TraceHandle, answer: Any) -> None:
    span.update(
        model=answer.model,
        usage_details=usage_details(answer.llm_meta or {}),
        metadata=answer_metadata(answer),
        output=handle.text(answer.answer),
    )
