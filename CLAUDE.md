# Claude Code Project Instructions - Medical RAG

## Mission
Build a reproducible, modular, client-demo-ready medical knowledge RAG platform using approved public documents. Demonstrate ingestion, OpenSearch dense+sparse hybrid retrieval, grounded answers with citations, evaluation, and operational performance.

## Decisions already made
- Domain: medical/healthcare information; not a diagnosis or treatment system.
- Backend: Python + FastAPI.
- Search/index store: OpenSearch.
- Retrieval: hybrid dense + sparse (exact sparse method remains open).
- RAG evaluation: Ragas.
- Development and handoff: Claude Code and a shareable GitHub repository.
- Measure retrieval quality, end-to-end P50/P95 latency, throughput and component timings.
- Langfuse is provisional for observability until its name is confirmed.

## Decisions NOT yet made
Exact corpus and licenses; chunking method; embedding model; vector algorithm; BM25 vs learned sparse; fusion method; reranker; generation LLM; UI; hosting; hardware; latency SLOs and scale targets. Consult docs/DECISION_LOG.md. Never silently convert a proposed default into a final architectural decision. Use swappable interfaces and configuration where possible.

## Implementation rules
1. Work one milestone/task at a time. Begin with a short plan, make a small change, run verification, summarize evidence, then stop for review at decision gates.
2. Prefer a readable Python package and explicit typed interfaces over framework-heavy abstractions. Keep ingestion, chunking, embedding, retrieval, fusion, reranking, generation, evaluation and tracing separable.
3. Use Python type hints, Pydantic schemas for API contracts, structured logs, configuration via environment variables, and deterministic IDs for documents/chunks.
4. Write unit tests for parsing and data models, integration tests for OpenSearch operations, and representative end-to-end smoke tests. Mock expensive external model calls in CI.
5. Pin dependency and container versions at implementation time; document versions actually tested. Avoid undocumented external services and unreviewed dependency sprawl.
6. No secrets or credentials in Git. Provide .env.example, .gitignore, and clearly identified local-only settings. Do not force-push or deploy without explicit authorization.
7. Use publicly redistributable test fixtures only after checking each source's actual reuse conditions. Preserve URL, title, document version/date, license, and chunk provenance. Exclude patient-identifiable data entirely.
8. Do not invent clinical citations. Return retrieved source passage identifiers and URLs. If evidence is insufficient, abstain clearly. Document that this is an informational demo, not medical advice.
9. Avoid logging sensitive query text or medical content by default. If full text tracing is enabled for synthetic demo inputs, document the opt-in, retention and redaction policy.
10. Keep a clear list of pending decisions; record explicit choices and benchmark evidence in docs/DECISION_LOG.md.

## Working protocol for each task
- Read docs/EXECUTION_CHECKLIST.md and docs/DECISION_LOG.md.
- State scope, touched modules, risks, and checks to run.
- Implement only the approved scope; add or update tests and docs.
- Report commands run and outcomes, unresolved limitations, and next small task.
- Do not claim a feature passes unless tests actually ran.

## Acceptance criteria for the first milestone
- Repository structure created with a basic FastAPI app and /health endpoint.
- Dependency management, formatting/linting, tests and CI wired up and executable.
- Config validation and .env.example available, no secrets committed.
- Docker Compose with a **local-development-only** OpenSearch service, pinned to an explicitly chosen version, plus readiness check; no claim of production hardening.
- Architecture overview, README and decision log completed.
- Unit tests and an API smoke test pass; commands and results recorded.
- Do not implement medical embedding, model selection, reranking, or LLM generation during this milestone.

## Documents
- docs/EXECUTION_CHECKLIST.md - phases, deliverables and exit criteria.
- docs/DECISION_LOG.md - confirmed, pending and proposed experiment decisions.
- docs/FIRST_CLAUDE_PROMPT.md - paste into Claude Code to begin.
