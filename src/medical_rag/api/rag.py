"""/search and /ask endpoints. Query text is not logged unless MEDRAG_LOG_QUERY_TEXT=true (CLAUDE.md rule 9)."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from medical_rag.core.config import Settings, get_settings
from medical_rag.observability import rag_trace
from medical_rag.observability.tracing import get_tracer
from medical_rag.search.client import make_client
from medical_rag.search.planning import plan_for
from medical_rag.search.retrieval import Retriever

log = logging.getLogger("medical_rag.api")
DISCLAIMER = "Informational demo over public documents. Not medical advice; not a diagnosis or treatment system."
SOURCES = Literal["dailymed", "pmc_oa", "nih_livertox", "fda_drugsatfda"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    k: int = Field(default=8, ge=1, le=50)
    sources: list[SOURCES] | None = None


class ChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    document_version: str | None = None
    title: str
    section_path: str | None = None
    section_role: str | None = None
    evidence_type: str | None = None
    source: str
    source_reference: str
    medicine_names: list[str] = []
    kind: str | None = None
    ordinal: int | None = None
    token_count: int | None = None
    overlap_tokens: int | None = None
    notices: list[str] = []
    split_levels: list[str] = []
    prefix: str = ""
    text: str


class Passage(ChunkOut):
    rank: int
    score: float
    component_ranks: dict[str, int]


class ChunkDetail(BaseModel):
    chunk: ChunkOut
    previous: ChunkOut | None
    next: ChunkOut | None
    document_chunks: int


class InfoResponse(BaseModel):
    index: str | None
    chunks: int | None
    chunks_by_source: dict[str, int]
    embedding_model: str
    llm_configured: bool
    llm_model: str
    disclaimer: str = DISCLAIMER


class SearchResponse(BaseModel):
    passages: list[Passage]
    timings_ms: dict[str, float]
    disclaimer: str = DISCLAIMER


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=8, ge=1, le=20)
    sources: list[SOURCES] | None = None


class CitationOut(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    title: str
    section_path: str
    source_reference: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    abstain_reason: str | None
    coverage: str | None = None  # prompt v4: full / related / none
    citations: list[CitationOut]
    retrieved: list[Passage]
    model: str
    timings_ms: dict[str, float]
    disclaimer: str = DISCLAIMER


@lru_cache
def _embedder():
    from medical_rag.embedding.config import BGE_QUERY_INSTRUCTION, EmbeddingConfig
    from medical_rag.embedding.st_embedder import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(EmbeddingConfig(query_instruction=BGE_QUERY_INSTRUCTION))


@lru_cache
def _llm():
    from medical_rag.generation.claude import ClaudeLLM

    return ClaudeLLM(get_settings())


@lru_cache
def _search_client():
    return make_client(get_settings())  # one client per process; connections are pooled and reused


def get_retriever(settings: Annotated[Settings, Depends(get_settings)]) -> Retriever:
    return Retriever(_search_client(), _embedder(), index=settings.index_alias)


RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
router = APIRouter()


def _chunk(source: dict) -> dict:
    out = {k: v for k, v in source.items() if k in ChunkOut.model_fields and v is not None}
    names = out.get("medicine_names")
    if isinstance(names, str):
        out["medicine_names"] = [names]
    return out


def _passages(result) -> list[Passage]:
    return [
        Passage(rank=i, score=h.score, component_ranks=h.ranks, **_chunk(h.source))
        for i, h in enumerate(result.hits, start=1)
    ]


@router.get("/info", response_model=InfoResponse)
def info(settings: Annotated[Settings, Depends(get_settings)]):
    """Which index is served, how many chunks per source, and whether answers are available (no secrets)."""
    from medical_rag.embedding.config import EmbeddingConfig

    client = _search_client()
    index, count, by_source = None, None, {}
    try:
        index = next(iter(client.indices.get_alias(name=settings.index_alias)))
        body = {"size": 0, "aggs": {"s": {"terms": {"field": "source", "size": 20}}}}
        res = client.search(index=settings.index_alias, body=body)
        count = res["hits"]["total"]["value"]
        by_source = {b["key"]: b["doc_count"] for b in res["aggregations"]["s"]["buckets"]}
    except Exception as exc:  # noqa: BLE001 - info must not fail when the index is missing
        log.warning("info_unavailable", extra={"error": type(exc).__name__})
    return InfoResponse(
        index=index,
        chunks=count,
        chunks_by_source=by_source,
        embedding_model=EmbeddingConfig.model_fields["model_name"].default,
        llm_configured=settings.anthropic_api_key is not None,
        llm_model=settings.llm_model,
    )


@router.get("/chunks/{chunk_id}", response_model=ChunkDetail)
def chunk_detail(chunk_id: str, settings: Annotated[Settings, Depends(get_settings)]):
    """A chunk with its previous and next chunk in the same document, for checking boundaries and overlap."""
    from medical_rag.search.retrieval import RETURN_FIELDS

    client = _search_client()
    hits = client.search(
        index=settings.index_alias,
        body={"size": 1, "_source": RETURN_FIELDS, "query": {"term": {"chunk_id": chunk_id}}},
    )["hits"]["hits"]
    if not hits:
        raise HTTPException(status_code=404, detail="chunk not found")
    chunk = hits[0]["_source"]
    doc_filter = {"term": {"document_id": chunk["document_id"]}}

    def neighbour(ordinal: int) -> ChunkOut | None:
        body = {
            "size": 1,
            "_source": RETURN_FIELDS,
            "query": {"bool": {"filter": [doc_filter, {"term": {"ordinal": ordinal}}]}},
        }
        found = client.search(index=settings.index_alias, body=body)["hits"]["hits"]
        return ChunkOut(**_chunk(found[0]["_source"])) if found else None

    ordinal = chunk.get("ordinal") or 0
    total = client.count(index=settings.index_alias, body={"query": doc_filter})["count"]
    return ChunkDetail(
        chunk=ChunkOut(**_chunk(chunk)),
        previous=neighbour(ordinal - 1) if ordinal > 0 else None,
        next=neighbour(ordinal + 1),
        document_chunks=total,
    )


def _log(settings: Settings, event: str, text: str, **fields) -> None:
    extra = {"query_text": text} if settings.log_query_text else {"query_chars": len(text)}
    log.info(event, extra={**extra, **fields})


@router.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest, request: Request, retriever: RetrieverDep, settings: Annotated[Settings, Depends(get_settings)]
):
    tracer = get_tracer()
    trace = tracer.start(
        "search",
        user_id=getattr(request.state, "user_id", None),
        tags=rag_trace.tags(settings),
        metadata={"query_chars": len(req.query)},
        input=tracer.text(req.query),
    )
    try:
        filters = {"source": req.sources} if req.sources else None
        with trace.span("retrieve", as_type="retriever") as span:
            plan = plan_for(req.query, settings, req.mode)
            result = retriever.search(
                req.query, mode=req.mode, k=req.k, candidates=settings.retrieval_candidates, filters=filters, plan=plan
            )
            span.update(
                metadata=rag_trace.retrieval_metadata(result, plan, mode=req.mode, k=req.k),
                output=rag_trace.retrieval_output(trace, result),
            )
        _log(settings, "search", req.query, mode=req.mode, hits=len(result.hits), **result.timings_ms)
        trace.update(output={"passages": len(result.hits)}, metadata={"total_ms": result.timings_ms.get("total_ms")})
        return SearchResponse(passages=_passages(result), timings_ms=result.timings_ms)
    finally:
        trace.end()


@router.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest, request: Request, retriever: RetrieverDep, settings: Annotated[Settings, Depends(get_settings)]
):
    from medical_rag.generation.answerer import answer_question
    from medical_rag.generation.claude import LLMError

    start = time.perf_counter()
    tracer = get_tracer()
    trace = tracer.start(
        "ask",
        user_id=getattr(request.state, "user_id", None),
        tags=rag_trace.tags(settings),
        metadata={"question_chars": len(req.question)},
        input=tracer.text(req.question),
    )
    try:
        return _ask(req, retriever, settings, trace, start, answer_question, LLMError)
    finally:
        trace.end()


def _ask(req, retriever, settings, trace, start, answer_question, LLMError) -> AskResponse:  # noqa: ANN001
    filters = {"source": req.sources} if req.sources else None
    with trace.span("retrieve", as_type="retriever") as span:
        plan = plan_for(req.question, settings)
        result = retriever.search(
            req.question, k=req.k, candidates=settings.retrieval_candidates, filters=filters, plan=plan
        )
        span.update(
            metadata=rag_trace.retrieval_metadata(result, plan, mode="hybrid", k=req.k),
            output=rag_trace.retrieval_output(trace, result),
        )
    with trace.span("compose", as_type="generation", model=settings.llm_model) as span:
        try:
            answer = answer_question(_llm(), req.question, [h.source for h in result.hits])
        except LLMError as exc:
            span.update(level="ERROR", status_message=str(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        rag_trace.record_answer(span, trace, answer)
    timings = {
        **{f"retrieval_{k}": v for k, v in result.timings_ms.items()},
        **answer.timings_ms,
        "total_ms": round((time.perf_counter() - start) * 1000, 2),
    }
    _log(settings, "ask", req.question, abstained=answer.abstained, citations=len(answer.citations), **timings)
    trace.update(
        output=trace.text(answer.answer) or {"answer_chars": len(answer.answer), "citations": len(answer.citations)},
        metadata={"total_ms": timings["total_ms"], "coverage": answer.llm_meta.get("coverage")},
    )
    return AskResponse(
        answer=answer.answer,
        abstained=answer.abstained,
        abstain_reason=answer.abstain_reason,
        coverage=answer.llm_meta.get("coverage"),
        citations=[CitationOut(**c.__dict__) for c in answer.citations],
        retrieved=_passages(result),
        model=answer.model,
        timings_ms=timings,
    )
