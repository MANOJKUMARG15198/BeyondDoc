# Medical RAG — Progress (2026-11-03)

## Summary
- **Phase 1 (data collection):** pipeline working; **Batch 1 = 100 documents collected and verified**.
- **Phase 0 (foundation):** built, tested and committed (`e26b462`, local `main`, no remote).
- **Next:** core RAG decisions, one at a time (chunking → embedding → index → retrieval → LLM → UI).

## Data collection

| Step | Result |
|---|---|
| Sources checked against official terms | DailyMed, openFDA, Drugs@FDA, PMC OA, LiverTox (WHO optional). PMC OA Web Service retired Aug 2026, so the AWS bucket is used |
| openFDA bulk export (2026-11-02) | 16 files, 1.8 GB, SHA-256 verified |
| Label selection | 262,904 labels → 12,928 duplicate groups → **910 candidates** (650 Rx / 260 OTC, 452 drug classes) |
| Batch 1 download | **100/100 documents**, 181 files, 290 MB, 0 failures, **2,214 PDF pages** |

**Batch 1:** 70 DailyMed labels (30 popular Rx, 20 boxed-warning Rx, 20 OTC) · 10 FDA approval letters · 15 PMC CC BY articles · 5 LiverTox chapters, all built around the same set of drugs.

**Key decisions (D1–D11):** non-commercial use with UI credits · raw data out of Git · brand-first representative label · grouping by ingredient + route + form + Rx/OTC · category caps · 500 Rx / 200 OTC · hybrid class + popularity ranking. Details are in `docs/data/SOURCE_REGISTER.md`.

## Foundation

| Area | Done |
|---|---|
| Stack | Python 3.11 · OpenSearch 3.8.0 (Docker, local only) · pip pinned requirements |
| Code | `src/medical_rag/` with 9 module boundaries · typed `MEDRAG_*` settings · JSON logs · FastAPI `/health` |
| Quality | ruff · 8 unit + 3 integration tests (all pass) · GitHub Actions CI (not yet run; no remote) |
| Docs | README · architecture · decision log · checklist · source register |

## Open items
- **User decisions pending:** D-2 chunking, D-3 embedding, D-4 vector index, D-5 sparse, D-6 fusion/reranker, D-7 LLM, D-8 UI/evaluation.
- OTC list quality · PMC licenses beyond CC BY · case reports · repository LICENSE · GitHub remote.
- Later: Batches 2–4 (900 more documents), older label versions, excipient-flag review.
- Constraints: no GPU · Docker capped at 6 GB · C: has 23 GB free.
