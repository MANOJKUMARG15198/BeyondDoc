# Medical RAG - Execution Checklist

Status date: 2026-11-03. `[x]` means agreed in discussion, NOT implemented in code. All implementation checkboxes start unchecked.

## A. Confirmed scope
- [x] Public medical/healthcare knowledge domain.
- [x] Python + FastAPI backend.
- [x] OpenSearch retrieval and vector store.
- [x] Dense + sparse hybrid retrieval architecture.
- [x] Ragas for RAG evaluation.
- [x] Claude Code for end-to-end implementation.
- [x] GitHub repository as handoff artifact.
- [x] Test retrieval quality, P50/P95 latency, and concurrency/scale behavior.
- [ ] Confirm whether "Lang views" meant Langfuse (working assumption).

## B. Outstanding architecture decisions - do not choose silently
- [ ] Select medical source(s), redistribution/license policies, document types, and refresh frequency.
- [ ] Choose Python and OpenSearch versions for reproducible builds.
- [ ] Compare chunk granularity and algorithms (section-aware, recursive, semantic, parent-child).
- [ ] Choose embedding model(s), dimensions, tokenizer and serving mode.
- [ ] Compare Flat/exact, HNSW and IVF when compatible with deployed OpenSearch version/engine.
- [ ] Set sparse strategy: BM25 baseline and/or neural sparse embeddings.
- [ ] Select fusion: RRF or normalized weighted scores.
- [ ] Decide if reranking is warranted and choose model.
- [ ] Choose answer-generation LLM and inference endpoint.
- [ ] Select frontend/demo interface.
- [ ] Define hardware, deployment, target corpus volume and workload.
- [ ] Define evaluation dataset and pass/fail thresholds.

## C. Implementation plan

### Phase 0 - Repo foundation (start here)
- [~] Create Git repository structure, README, license and contribution guidance. *(2026-11-03: local repo + README done; LICENSE and CONTRIBUTING pending — license choice is the user's)*
- [x] Set up Python package, FastAPI app, `/health`, Pydantic settings and .env.example. *(2026-11-03)*
- [~] Add tests, lint/format/type-check strategy, and GitHub Actions CI. *(ruff + pytest + CI workflow; no type checker per ADR-018; CI not yet run on GitHub — no remote)*
- [x] Add Compose-based local OpenSearch with pinned version, health check and clear local-only security notes. *(OpenSearch 3.8.0, healthy locally)*
- [x] Add docs/architecture.md, ADR log, data-provenance template. *(provenance: docs/data/SOURCE_REGISTER.md + data/manifests)*
- [ ] Verify a fresh clone can run installation, health tests, and CI commands. *(needs a first commit)*

**Exit gate:** Clean clone works; `/health` responds; unit and CI checks pass; no data models or embeddings selected by assumption.

### Phase 1 - Ingestion and document governance
- [ ] Obtain a small permitted-to-use public medical document subset.
- [ ] Build source connector(s), parsing/normalization, metadata extraction and deduplication.
- [ ] Use stable document IDs, source URLs, revision timestamps, licenses and content hashes.
- [ ] Implement repeatable ingestion, retry/error reporting and safe incremental updates.
- [ ] Add synthetic/redistributable fixture corpus and ingestion tests.

**Exit gate:** Repeated ingestion doesn't duplicate content; provenance preserved; malformed sources handled.

### Phase 2 - Chunking, embeddings and OpenSearch indexes
- [ ] Implement chunker and embedding model interfaces with configuration options.
- [ ] Create OpenSearch text fields and vector mapping based on chosen model dimension.
- [ ] Implement and test the selected vector index method.
- [ ] Index chunks with stable IDs, parent document ID and source metadata.
- [ ] Validate indexing/upsert/delete and index rebuilding from a versioned dataset.

**Exit gate:** Known corpus yields deterministic chunk IDs, indexed text and queryable vectors.

### Phase 3 - Dense + sparse retrieval and answer API
- [ ] Implement sparse and dense retrieval independently.
- [ ] Combine results through a documented search pipeline and configurable fusion.
- [ ] Record scores, chunk IDs, source metadata and retrieval timings.
- [ ] Implement `/query` with sources; integrate a configurable generator once chosen.
- [ ] Add explicit insufficient-evidence and error-handling behavior.

**Exit gate:** Synthetic reference queries retrieve expected evidence and responses cite traceable sources.

### Phase 4 - Evaluation and observability
- [ ] Create versioned question set: factual, synonym, multi-document, no-answer and stale-source cases.
- [ ] Implement Recall@K, MRR/nDCG and Ragas faithfulness/relevance metrics as applicable.
- [ ] Add Langfuse (if confirmed) or adapter-based tracing.
- [ ] Record per-stage timings, model/token usage and experiment configurations.
- [ ] Compare dense, sparse and hybrid against an identical corpus/question set.

**Exit gate:** Repeatable quality report and trace evidence for representative queries.

### Phase 5 - Optimization, latency and scaling
- [ ] Experiment with chunking, embedding, HNSW/IVF/Flat as applicable, fusion and reranking.
- [ ] Build load-test scenarios with clear hardware and concurrency details.
- [ ] Report P50/P95 end-to-end latency and stage latencies, throughput and errors.
- [ ] Report indexing throughput, RAM/disk consumption and corpus size.
- [ ] Document accuracy-versus-latency and memory/cost trade-offs.

**Exit gate:** Report includes reproducible commands, versions, dataset size, hardware, QPS, error rate and metrics.

### Phase 6 - Client demo and GitHub handoff
- [ ] Provide demo script with citation, synonym, conflicting-source and no-answer examples.
- [ ] Build a simple usable demo UI or API collection as agreed.
- [ ] Document quickstart, config, architecture, security limits, troubleshooting and runbooks.
- [ ] Add sanitized sample data, reproducible benchmarks and architecture diagrams.
- [ ] Verify a fresh developer can follow README and reproduce the demo.
- [ ] Final security/license/code review before sharing the repository.

**Exit gate:** Reproducible public-safe demo and complete developer handoff.

## D. Quality and safety gates (every phase)
- [ ] Small scoped change; tests actually run and results recorded.
- [ ] No secrets, credentials, patient data or unlicensed copies in Git.
- [ ] Update README/architecture/decision records if behavior changes.
- [ ] Store source attribution and dates; never fabricate medical citations.
- [ ] Separate measured results from proposed performance targets.
- [ ] Obtain review before changing confirmed architecture decisions.
