# Architecture

Status 2026-11-05: local end-to-end system running (ingestion → chunking → embedding → hybrid search → grounded
answers → chat UI with sessions). Choices marked *demo default* are provisional; see `DECISION_LOG.md`.
The scaling target on Google Cloud (GKE) is a **concept only**: [SCALING_GCP_GKE.md](SCALING_GCP_GKE.md).

```
 public sources ──► scripts/phase1 (discover, select, download) ──► data/raw + data/manifests
                                                                        │
              ingestion: SPL, JATS, LiverTox NXML, FDA PDF parsers · normalization · batch-plan catalog
                                                                        │
              chunking: structure-aware recursive, Fill-mode overlap (demo 64 tokens), BERT-uncased counts
                                                                        │
              embedding: BAAI/bge-base-en-v1.5 (pinned, checksum-verified, CPU)
                                                                        │
              OpenSearch 3.8.0: one index, BM25 text + HNSW vectors (Lucene, cosine), alias medrag-chunks
                                                                        │
  React chat UI ──SSE──► FastAPI /api/sessions/{id}/messages
                           │  Redis: last turns, rate limit      ClickHouse: chat history (TTL 90 days)
                           ├─ follow-up rewrite (Claude) ─► hybrid retrieval (BM25 + dense, RRF) ─► k passages
                           └─ grounded answer (Claude Sonnet 5.5): cite passage ids or abstain
                                                                        │
              evaluation: hit/recall/MRR/nDCG, Ragas (Claude Opus 5.5 judge) · tracing: Langfuse (planned)
```

## Boundaries

| Module | Responsibility | Status |
|---|---|---|
| `core.config` | Typed `MEDRAG_*` settings, secrets as `SecretStr`, validated at startup | Implemented |
| `observability.logging` | JSON-lines logs; medical query text off by default | Implemented |
| `ingestion` | Parsers, normalization, provenance, catalog titles/medicine names | Implemented |
| `chunking` | Chunker, tokenizer, tables, validation (V1–V11), JSONL output | Implemented |
| `embedding` | `Embedder` interface, sentence-transformers implementation, resumable vector store | Implemented |
| `search` | Client, index mapping/build (alias swap), BM25/dense/hybrid retrieval with RRF | Implemented (demo defaults) |
| `generation` | Grounded answers with citation checks and abstention; Claude client | Implemented (demo default) |
| `chat` | Session store (ClickHouse / in-memory), short-term memory (Redis / in-memory), turn pipeline | Implemented |
| `evaluation` | Retrieval metrics, test set model, Ragas adapters | Implemented; runs pending review |
| `api` | `/health`, `/search`, `/ask`, `/info`, `/chunks/{id}`, `/api/sessions…` (SSE), UI at `/`, explorer at `/explorer` | Implemented |
| `frontend/` | React + Vite Beyond Doc chat UI, built into `api/static/app` | Implemented |

## Local development assumptions

- One machine (Windows 11, 4 cores / 8 threads, 15.8 GB RAM, **no GPU**). Docker is capped at 6 GB.
- Docker Compose (local only, bound to 127.0.0.1): OpenSearch 3.8.0 (security plugin disabled, 1 GB heap),
  Redis 7.4.11 (no persistence), ClickHouse 25.3 (password from `.env`). Not hardened.
- Secrets (Anthropic key, ClickHouse password) only in the gitignored `.env`.
- Corpus files stay outside Git (`data/`). Provenance (URL, timestamp, SHA-256, license basis) is in `data/manifests/`.

## Identifiers

- Logical document: `<source>:<source_document_id>`, e.g. `dailymed:<SETID>`, `pmc_oa:PMC123.1`, `fda_drugsatfda:NDA020702:<doc id>`.
- Chunk ID: SHA-256 (32 hex) over canonical JSON of document, version, section and content spans; `config_id` hashes
  every chunking setting, the tokenizer identity and `CHUNKER_VERSION`.
- Index name: `medrag-chunks-<chunk config_id>-<embedder_id>`, so a different chunking or model never mixes into an index.
- Chat: anonymous user id (cookie), `session_id` and `message_id` are random UUID hex.
