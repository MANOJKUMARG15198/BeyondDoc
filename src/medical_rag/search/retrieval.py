"""BM25, dense and hybrid retrieval with client-side RRF (demo defaults ADR-013/014, provisional, swappable).

Fusion runs in Python rather than an OpenSearch search pipeline so each component's ranks and timings are visible.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
from opensearchpy import OpenSearch

from medical_rag.search.index import ALIAS
from medical_rag.search.planner import CoverageUnit, RetrievalPlan
from medical_rag.search.selection import apply_coverage_floor, apply_identity_guard, apply_noise_demotion, is_noise

Mode = Literal["bm25", "dense", "hybrid"]
RETURN_FIELDS = [
    "chunk_id",
    "document_id",
    "document_version",
    "source",
    "source_reference",
    "title",
    "section_path",
    "section_role",
    "medicine_names",
    "evidence_type",
    "kind",
    "ordinal",
    "token_count",
    "overlap_tokens",
    "notices",
    "split_levels",
    "prefix",
    "text",
]


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass
class Hit:
    chunk_id: str
    score: float
    source: dict[str, Any]
    ranks: dict[str, int] = field(default_factory=dict)  # 1-based rank per component


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float]
    plan_reasons: tuple[str, ...] = ()
    selection: dict[str, str] = field(default_factory=dict)  # chunk_id -> reason (e.g. demoted_other_medicine)


def rrf(rankings: Mapping[str, Sequence[str]], k: int = 60) -> list[tuple[str, float, dict[str, int]]]:
    """Reciprocal rank fusion: score = sum 1 / (k + rank). Ties broken by chunk id for determinism."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(chunk_id, {})[name] = rank
    order = sorted(scores, key=lambda c: (-scores[c], c))
    return [(c, scores[c], ranks[c]) for c in order]


def _filters(filters: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    out = []
    for name, value in (filters or {}).items():
        values = value if isinstance(value, list | tuple) else [value]
        out.append({"terms": {name: list(values)}})
    return out


def bm25_body(
    query: str, size: int, filters: Mapping[str, Any] | None = None, should: Sequence[dict[str, Any]] = ()
) -> dict[str, Any]:
    """`should` adds soft boosts (Retrieval v1.2); without it the body is exactly the original one."""
    clause: dict[str, Any] = {"must": [{"match": {"search_text": query}}], "filter": _filters(filters)}
    if should:
        clause["should"] = list(should)
    return {"size": size, "_source": RETURN_FIELDS, "query": {"bool": clause}}


def medicine_boosts(plan: RetrievalPlan) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if plan.boost_documents:
        out.append({"terms": {"document_id": list(plan.boost_documents), "boost": plan.document_boost}})
    if plan.boost_aliases:
        out.append({"match": {"medicine_names": {"query": " ".join(plan.boost_aliases), "boost": plan.alias_boost}}})
    return out


def section_boosts(plan: RetrievalPlan) -> list[dict[str, Any]]:
    """Soft section preference from the question's intents (Retrieval v1.2, Step 5)."""
    out: list[dict[str, Any]] = [
        {"match_phrase": {"section_path": {"query": p, "boost": plan.section_boost}}} for p in plan.section_phrases
    ]
    out += [{"term": {"section_role": {"value": r, "boost": plan.section_boost}}} for r in plan.section_roles]
    return out


def dense_body(vector: np.ndarray, size: int, filters: Mapping[str, Any] | None = None) -> dict[str, Any]:
    knn: dict[str, Any] = {"vector": [float(x) for x in vector], "k": size}
    if filters:
        knn["filter"] = {"bool": {"filter": _filters(filters)}}
    return {"size": size, "_source": RETURN_FIELDS, "query": {"knn": {"embedding": knn}}}


class Retriever:
    def __init__(self, client: OpenSearch, embedder: QueryEmbedder, index: str = ALIAS, rrf_k: int = 60) -> None:
        self.client, self.embedder, self.index, self.rrf_k = client, embedder, index, rrf_k

    def _run(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        return self.client.search(index=self.index, body=body)["hits"]["hits"]

    def _coverage_branches(
        self, query: str, vector: np.ndarray, plan: RetrievalPlan, filters: Mapping[str, Any] | None
    ) -> tuple[dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]], list[str]]:
        """One filtered hybrid search per coverage unit, run concurrently on the shared index with the same query
        vector. A branch that errors or exceeds plan.branch_timeout_s is skipped (logged), never fatal."""
        from concurrent.futures import ThreadPoolExecutor, wait

        def run(unit: CoverageUnit) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            restricted = {**(filters or {}), "document_id": list(unit.documents)}
            return (
                self._run(bm25_body(query, plan.branch_size, restricted, section_boosts(plan))),
                self._run(dense_body(vector, plan.branch_size, restricted)),
            )

        executor = ThreadPoolExecutor(max_workers=len(plan.coverage))
        futures = {executor.submit(run, unit): unit for unit in plan.coverage}
        done, _ = wait(futures, timeout=plan.branch_timeout_s)
        executor.shutdown(wait=False, cancel_futures=True)
        results, failed = {}, []
        for future, unit in futures.items():
            if future in done and future.exception() is None:
                results[unit.key] = future.result()
            else:
                failed.append(unit.key)
        return results, failed

    def search(
        self,
        query: str,
        *,
        mode: Mode = "hybrid",
        k: int = 10,
        candidates: int = 50,
        filters: Mapping[str, Any] | None = None,
        plan: RetrievalPlan | None = None,
    ) -> SearchResult:
        """`plan` (Retrieval v1.2, hybrid only): soft BM25 boosts, an extra medicine-filtered dense list in RRF, and
        the post-fusion identity guard. Without a plan, or with the standard plan, behaviour is unchanged."""
        active = plan is not None and not plan.is_standard
        if active and mode != "hybrid":
            raise ValueError("a retrieval plan is only supported in hybrid mode")
        timings: dict[str, float] = {}
        start = time.perf_counter()
        pool = candidates if mode == "hybrid" else k
        lists: dict[str, list[dict[str, Any]]] = {}
        if mode in ("bm25", "hybrid"):
            t = time.perf_counter()
            should = [*medicine_boosts(plan), *section_boosts(plan)] if active else ()
            lists["bm25"] = self._run(bm25_body(query, pool, filters, should))
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000
        if mode in ("dense", "hybrid"):
            t = time.perf_counter()
            vector = self.embedder.embed_query(query)
            timings["embed_query_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            lists["dense"] = self._run(dense_body(vector, pool, filters))
            timings["dense_ms"] = (time.perf_counter() - t) * 1000
            if active and plan.medicine_dense_documents:  # same query vector, restricted to the medicine's documents
                t = time.perf_counter()
                restricted = {**(filters or {}), "document_id": list(plan.medicine_dense_documents)}
                lists["dense_medicine"] = self._run(dense_body(vector, pool, restricted))
                timings["dense_medicine_ms"] = (time.perf_counter() - t) * 1000
        branches: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
        failed: list[str] = []
        if active and plan.coverage and mode == "hybrid":
            t = time.perf_counter()
            branches, failed = self._coverage_branches(query, vector, plan, filters)
            for key, (b, d) in branches.items():
                lists[f"cov:{key}:bm25"], lists[f"cov:{key}:dense"] = b, d
            timings["coverage_ms"] = (time.perf_counter() - t) * 1000

        sources = {h["_id"]: h["_source"] for hits in lists.values() for h in hits}
        selection: dict[str, str] = {}
        if mode == "hybrid":
            t = time.perf_counter()
            fused = rrf({name: [h["_id"] for h in hits] for name, hits in lists.items()}, self.rrf_k)
            if not active:
                fused = fused[:k]
            timings["fusion_ms"] = (time.perf_counter() - t) * 1000
            hits = [Hit(c, s, sources[c], r) for c, s, r in fused]
            if active and plan.guard is not None:
                t = time.perf_counter()
                hits, selection = apply_identity_guard(hits, plan.guard.medicine_documents, plan.guard.max_other)
                timings["selection_ms"] = (time.perf_counter() - t) * 1000
            if active and plan.demote_noise:
                hits, reasons_noise = apply_noise_demotion(hits, plan.max_noise)
                selection |= reasons_noise
            if active and branches:
                units = []
                for unit in plan.coverage:
                    if unit.key in branches:
                        b, d = branches[unit.key]
                        order = rrf({"b": [h["_id"] for h in b], "d": [h["_id"] for h in d]}, self.rrf_k)
                        ids = [c for c, _, _ in order]
                        if plan.demote_noise:  # coverage never reserves a place for a noise passage
                            ids = [c for c in ids if not is_noise(sources[c])]
                        units.append((unit.key, ids, {h["_id"] for h in b}))
                floor = plan.coverage_floor or max(1, k // (len(plan.coverage) + 1))
                hits, reasons_cov = apply_coverage_floor(hits, units, floor, k)
                selection |= reasons_cov
            selection |= {f"branch_failed:{key}": "skipped" for key in failed}
            hits = hits[:k]
            kept = {h.chunk_id for h in hits}
            selection = {c: r for c, r in selection.items() if c in kept or c.startswith("branch_failed:")}
        else:
            ((name, raw),) = lists.items()
            hits = [Hit(h["_id"], float(h["_score"]), h["_source"], {name: i}) for i, h in enumerate(raw, start=1)]
        timings["total_ms"] = (time.perf_counter() - start) * 1000
        reasons = plan.reasons if plan is not None else ()
        return SearchResult(hits, {key: round(v, 2) for key, v in timings.items()}, reasons, selection)
