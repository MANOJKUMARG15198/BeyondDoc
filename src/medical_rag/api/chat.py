"""Chat API: per-user sessions (anonymous cookie id), history, and a streaming (SSE) message endpoint.

Question text is stored as chat history in the local ClickHouse (retention MEDRAG_CHAT_RETENTION_DAYS). It is not
written to logs unless MEDRAG_LOG_QUERY_TEXT=true. Local demo for synthetic questions only; not medical advice.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from medical_rag.api.rag import SOURCES, _embedder, _llm, _search_client
from medical_rag.chat.service import ChatService, message_view, session_view
from medical_rag.core.config import Settings, get_settings

log = logging.getLogger("medical_rag.chat")
COOKIE = "medrag_uid"
UID = re.compile(r"^[a-f0-9]{32}$")
router = APIRouter(prefix="/api")


def assign_user(request: Request) -> str | None:
    """Return a new id when the request has no valid cookie (the middleware then sets it)."""
    uid = request.cookies.get(COOKIE, "")
    if UID.match(uid):
        request.state.user_id = uid
        return None
    request.state.user_id = uuid.uuid4().hex
    return request.state.user_id


def current_user(request: Request) -> str:
    return request.state.user_id


@lru_cache
def _service() -> ChatService:
    from medical_rag.chat.memory import RedisMemory
    from medical_rag.chat.store import ClickHouseStore
    from medical_rag.observability import rag_trace
    from medical_rag.observability.tracing import get_tracer
    from medical_rag.search.planning import plan_for
    from medical_rag.search.retrieval import Retriever

    settings = get_settings()
    retriever = Retriever(_search_client(), _embedder(), index=settings.index_alias)
    return ChatService(
        ClickHouseStore.from_settings(settings),
        RedisMemory.from_settings(settings),
        retriever,
        _llm(),
        settings.chat_memory_turns,
        planner=(lambda query, mode: plan_for(query, settings, mode)) if settings.retrieval_plan != "none" else None,
        tracer=get_tracer(),
        trace_tags=rag_trace.tags(settings),
    )


def get_chat_service() -> ChatService:
    return _service()


UserDep = Annotated[str, Depends(current_user)]
ServiceDep = Annotated[ChatService, Depends(get_chat_service)]


class NewSession(BaseModel):
    title: str = Field(default="New consultation", max_length=120)


class RenameSession(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class NewMessage(BaseModel):
    content: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=8, ge=1, le=20)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    sources: list[SOURCES] | None = None


def _owned(service: ChatService, user: str, session_id: str):
    session = service.store.get_session(user, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat not found")
    return session


@router.get("/me")
def me(user: UserDep, settings: Annotated[Settings, Depends(get_settings)]):
    return {"user": user[:8], "retention_days": settings.chat_retention_days}


@router.get("/sessions")
def list_sessions(user: UserDep, service: ServiceDep):
    return {"sessions": [session_view(s) for s in service.store.list_sessions(user)]}


@router.post("/sessions", status_code=201)
def create_session(body: NewSession, user: UserDep, service: ServiceDep):
    return session_view(service.store.create_session(user, body.title))


@router.get("/sessions/{session_id}")
def get_session(session_id: str, user: UserDep, service: ServiceDep):
    session = _owned(service, user, session_id)
    messages = [message_view(m) for m in service.store.list_messages(session_id)]
    return {"session": session_view(session), "messages": messages}


@router.patch("/sessions/{session_id}")
def rename_session(session_id: str, body: RenameSession, user: UserDep, service: ServiceDep):
    from medical_rag.chat.store import now

    session = _owned(service, user, session_id)
    session.title, session.updated_at = body.title, now()
    service.store.update_session(session)
    return session_view(session)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, user: UserDep, service: ServiceDep):
    from medical_rag.chat.store import now

    session = _owned(service, user, session_id)
    session.deleted, session.updated_at = True, now()
    service.store.update_session(session)
    service.memory.replace(session_id, [])


def _sse(events: Iterator[dict]) -> Iterator[str]:
    try:
        for e in events:
            yield f"event: {e['event']}\ndata: {json.dumps(e['data'], default=str)}\n\n"
    except Exception as exc:  # noqa: BLE001 - the stream reports the failure instead of dropping the connection
        log.exception("chat_turn_failed")
        detail = str(exc) if exc.__class__.__name__ == "LLMError" else "The answer could not be generated."
        yield f"event: error\ndata: {json.dumps({'detail': detail})}\n\n"
    yield "event: done\ndata: {}\n\n"


@router.post("/sessions/{session_id}/messages")
def send_message(
    session_id: str,
    body: NewMessage,
    user: UserDep,
    service: ServiceDep,
    settings: Annotated[Settings, Depends(get_settings)],
):
    session = _owned(service, user, session_id)
    if not service.memory.allow(user):
        raise HTTPException(status_code=429, detail="Too many questions; please wait a minute.")
    extra = {"query_text": body.content} if settings.log_query_text else {"query_chars": len(body.content)}
    log.info("chat_turn", extra={"session": session_id[:8], "mode": body.mode, "k": body.k, **extra})
    filters = {"source": body.sources} if body.sources else None
    events = service.turn(session, body.content, k=body.k, mode=body.mode, filters=filters)
    return StreamingResponse(
        _sse(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
