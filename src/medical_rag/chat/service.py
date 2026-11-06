"""One chat turn: rewrite a follow-up into a standalone query, retrieve, answer with citations, persist.

`turn()` yields events (stage / sources / answer / error) that the API streams to the browser as SSE.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from medical_rag.chat.memory import ShortTermMemory
from medical_rag.chat.store import ChatMessage, ChatSession, SessionStore, new_id, now
from medical_rag.generation.answerer import LLM, answer_question
from medical_rag.observability import rag_trace
from medical_rag.observability.tracing import Tracer, usage_details

REWRITE_SYSTEM = """You turn a follow-up question into a standalone search query for a medicine document search.
Use the earlier conversation only to resolve references such as "it", "that drug" or "the dose". Keep medicine names,
doses and populations explicit. Do not answer the question. If it is already standalone, return it unchanged."""
REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"standalone_query": {"type": "string"}},
    "required": ["standalone_query"],
    "additionalProperties": False,
}
PASSAGE_FIELDS = (
    "chunk_id",
    "document_id",
    "title",
    "section_path",
    "section_role",
    "source",
    "source_reference",
    "medicine_names",
    "kind",
    "token_count",
)
STAGES = ("understand", "search", "compose")


def title_from(question: str, limit: int = 60) -> str:
    q = " ".join(question.split())
    return q if len(q) <= limit else q[: limit - 1].rsplit(" ", 1)[0] + "…"


def passage_view(rank: int, hit: Any) -> dict[str, Any]:
    src = hit.source
    view = {f: src.get(f) for f in PASSAGE_FIELDS}
    names = view.get("medicine_names")
    view["medicine_names"] = [names] if isinstance(names, str) else list(names or [])
    view.update(rank=rank, score=round(float(hit.score), 6), component_ranks=hit.ranks, text=src.get("text", ""))
    return view


def message_view(m: ChatMessage) -> dict[str, Any]:
    return {
        "message_id": m.message_id,
        "role": m.role,
        "content": m.content,
        "created_at": m.created_at.isoformat(),
        **m.payload,
    }


def session_view(s: ChatSession) -> dict[str, Any]:
    return {
        "session_id": s.session_id,
        "title": s.title,
        "created_at": s.created_at.isoformat(),
        "updated_at": s.updated_at.isoformat(),
        "message_count": s.message_count,
    }


class ChatService:
    def __init__(
        self,
        store: SessionStore,
        memory: ShortTermMemory,
        retriever: Any,
        llm: LLM,
        memory_turns: int,
        planner: Any = None,  # (query, mode) -> RetrievalPlan | None; Retrieval v1.2, off when None
        tracer: Tracer | None = None,  # ADR-008; no-op when None
        trace_tags: Sequence[str] = (),
    ):
        self.store, self.memory, self.retriever, self.llm, self.memory_turns = (
            store,
            memory,
            retriever,
            llm,
            memory_turns,
        )
        self.planner = planner
        self.tracer = tracer or Tracer()
        self.trace_tags = list(trace_tags)

    # ---------- history ----------
    def history(self, session_id: str) -> list[dict[str, str]]:
        turns = self.memory.recent(session_id)
        if turns is None:  # not cached (expired or new worker): rehydrate from ClickHouse
            msgs = self.store.list_messages(session_id)[-self.memory_turns :]
            turns = [{"role": m.role, "content": m.content} for m in msgs]
            self.memory.replace(session_id, turns)
        return turns

    def standalone_query(self, question: str, turns: Sequence[Mapping[str, str]]) -> str:
        return self._rewrite(question, turns)[0]

    def _rewrite(self, question: str, turns: Sequence[Mapping[str, str]]) -> tuple[str, dict[str, Any]]:
        if not turns:
            return question, {}
        convo = "\n".join(f"{t['role']}: {t['content'][:800]}" for t in turns)
        user = f"<earlier_conversation>\n{convo}\n</earlier_conversation>\n\n<follow_up>\n{question}\n</follow_up>"
        output, meta = self.llm.complete_json(REWRITE_SYSTEM, user, REWRITE_SCHEMA)
        return (output.get("standalone_query") or question).strip(), meta or {}

    # ---------- one turn ----------
    def turn(
        self,
        session: ChatSession,
        content: str,
        *,
        k: int = 8,
        mode: str = "hybrid",
        filters: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        trace = self.tracer.start(
            "chat_turn",
            user_id=session.user_id,
            session_id=session.session_id,
            tags=self.trace_tags,
            metadata={"question_chars": len(content), "mode": mode, "k": k},
            input=self.tracer.text(content),
        )
        try:
            t0 = time.perf_counter()
            timings: dict[str, float] = {}

            def stage(sid: str, status: str, detail: str = "") -> dict[str, Any]:
                return {"event": "stage", "data": {"stage": sid, "status": status, "detail": detail}}

            t = time.perf_counter()
            turns = self.history(session.session_id)  # before saving this question, so it is not its own history
            user_msg = ChatMessage(new_id(), session.session_id, session.user_id, "user", content, now())
            self.store.add_message(user_msg)
            yield {"event": "user_message", "data": message_view(user_msg)}

            yield stage("understand", "active")
            with trace.span("understand", as_type="generation" if turns else "span") as span:
                query, rewrite_meta = self._rewrite(content, turns)
                span.update(
                    model=self.llm.model if turns else None,
                    usage_details=usage_details(rewrite_meta),
                    metadata={"rewritten": bool(turns), "history_messages": len(turns), "query_chars": len(query)},
                    output=trace.text(query),
                )
            timings["understand_ms"] = round((time.perf_counter() - t) * 1000, 1)
            detail = "follow-up resolved" if turns else "new question"
            yield stage("understand", "done", detail)

            yield stage("search", "active")
            with trace.span("retrieve", as_type="retriever") as span:
                plan = self.planner(query, mode) if self.planner else None
                extra = {"plan": plan} if plan is not None else {}
                result = self.retriever.search(query, mode=mode, k=k, candidates=50, filters=filters, **extra)
                span.update(
                    metadata=rag_trace.retrieval_metadata(result, plan, mode=mode, k=k),
                    output=rag_trace.retrieval_output(trace, result),
                )
            timings.update({f"retrieval_{key}": v for key, v in result.timings_ms.items()})
            passages = [passage_view(i, h) for i, h in enumerate(result.hits, start=1)]
            yield stage("search", "done", f"{len(passages)} passages · {mode}")
            yield {"event": "sources", "data": {"query": query, "passages": passages}}

            yield stage("compose", "active")
            with trace.span("compose", as_type="generation", model=self.llm.model) as span:
                answer = answer_question(self.llm, content, [h.source for h in result.hits], turns)
                rag_trace.record_answer(span, trace, answer)
            timings.update(answer.timings_ms)
            timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            cited = [c.source_id for c in answer.citations]
            yield stage("compose", "done", f"{len(cited)} cited" if not answer.abstained else "abstained")

            payload = {
                "abstained": answer.abstained,
                "abstain_reason": answer.abstain_reason,
                "coverage": answer.llm_meta.get("coverage"),
                "cited": cited,
                "sources": [{**p, "text": p["text"][:1500]} for p in passages],
                "query": query,
                "mode": mode,
                "k": k,
                "model": answer.model,
                "timings_ms": timings,
            }
            bot_msg = ChatMessage(
                new_id(), session.session_id, session.user_id, "assistant", answer.answer, now(), payload
            )
            self.store.add_message(bot_msg)
            self.memory.append(
                session.session_id,
                [{"role": "user", "content": content}, {"role": "assistant", "content": answer.answer}],
            )
            if session.message_count == 0 or session.title == "New consultation":
                session.title = title_from(content)
            session.message_count += 2
            session.updated_at = now()
            self.store.update_session(session)
            yield {"event": "answer", "data": {"message": message_view(bot_msg), "session": session_view(session)}}
            trace.update(
                output=trace.text(answer.answer)
                or {"answer_chars": len(answer.answer), "citations": len(answer.citations)},
                metadata={"total_ms": timings.get("total_ms"), "coverage": answer.llm_meta.get("coverage")},
            )
        finally:
            trace.end()
