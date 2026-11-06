# Phase 4 — Reranking Experiment Report (Steps 1–5, MiniLM arms)

Date: 2026-11-05 · Status: development-only experiment; nothing deployed · Author: assistant, reviewed with the project owner

Scope: v2.0 dev set (75 questions, 67 answerable) from `data/eval/v2/frozen/retrieval_v2.0_split75-75.jsonl`
(75/75 split variant; holdout untouched). Covered: Step 1 audit, candidate extraction, offline arms A–C with
MiniLM-L6. **Not yet done:** MedCPT (arms D/E), concurrent latency validation (Step 6), end-to-end Ragas (Step 7).
Evaluation dataset, corpus, index and retrieval settings unchanged.

## 1. Baseline architecture

| Component | Configuration |
|---|---|
| Corpus / index | 5,815 chunks, 100 documents (DailyMed, PMC, LiverTox, FDA); OpenSearch 3.8.0, index `medrag-chunks-f6fab776e222268a-1bbc080536298982`, alias `medrag-chunks` |
| BM25 | `match` on `search_text` (prefix + newline + body), `english` analyzer, default BM25 parameters; **`medicine_names` not queried or filtered** |
| Dense | bge-base-en-v1.5 (768-d, CLS, L2-normalised), query instruction prefix, 512-token limit; HNSW Lucene cosine (m=16, ef_construction=128, ef_search=100) |
| Fusion | Client-side RRF (k=60); raw scores discarded; ties by chunk id |
| Candidates / final | 50 + 50 → 56–100 fused unique chunks; top **8** to the LLM |
| Dedup / filters | No dedup (0 duplicates observed); optional `source` filter; **no medicine-identity filter** |
| Generation | Claude Sonnet 5.5 (effort medium), validated citations, abstention |
| Instrumentation | `embed_query_ms`, `bm25_ms`, `dense_ms`, `fusion_ms`, `llm_ms`, `total_ms`; chat path adds an LLM follow-up rewrite; no token streaming (no TTFT) |

Relevant code: `src/medical_rag/search/index.py`, `src/medical_rag/search/retrieval.py`,
`src/medical_rag/api/rag.py`, `src/medical_rag/chat/service.py`.

## 2. Candidate retrieval analysis

- Candidates extracted with `scripts/phase4/extract_candidates.py` (retrieval only) to
  `data/processed/eval/candidates/retrieval_v2.0_split75-75_dev_candidates.jsonl` (+ manifest).
  **Deterministic:** top 8 identical, in order, to the v2 dev Ragas run for all 75 questions.
- The two "candidate" figures differ:
  - *Gold in pool 100%*: ≥1 gold chunk of the question in the pool — 67/67 questions.
  - *Pool recall 0.985*: mean per question of the share of evidence items in the pool. q076 and q077 each have one of
    two evidence items absent from the whole pool.

| Depth | Evidence recall | Questions with all evidence | Multi-document |
|---|---:|---:|---:|
| Top 8 | 0.873 | 56/67 | 0.31 |
| Top 20 | 0.925 | 60/67 | 0.50 |
| Top 40 | 0.978 | 64/67 | 0.81 |
| Full pool (≤100) | 0.985 | 65/67 | — |

## 3. Reranker model comparison

| | MiniLM-L6 (tested) | MedCPT Cross-Encoder (not tested) |
|---|---|---|
| Identity | `cross-encoder/ms-marco-MiniLM-L6-v2` @ `233902d25c440f23af6f7d6e94d2946bac0bee0a`; `model.safetensors` sha256 `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`; Apache-2.0 | `ncbi/MedCPT-Cross-Encoder`; license not yet checked; not downloaded |
| Architecture | BERT 6 layers, hidden 384, ~22M params (verified from config) | PubMedBERT-base, 12 layers, ~110M params (documentation) |
| Input limit | 512 tokens; vocabulary byte-identical to bge (token budgets exact) | 512 tokens; PubMed vocabulary |
| Domain | MS MARCO web search | PubMed search logs (titles/abstracts, not drug labels) |
| CPU (4-core laptop) | **Measured 3.6 s P50 for ~23 pairs** | ~5× compute → likely ≥15 s/question |
| Weights / load | ~90 MB; cold load 0.5–1.0 s; RSS not measured | ~440 MB |

## 4. Token budget and evidence-integrity design

- 20% of chunks (1,190/5,815) and 15.2% of retrieved top-8 passages would exceed 512 tokens with the question.
- `src/medical_rag/search/rerank.py`:
  - no silent truncation — over-long passages are windowed;
  - every window carries the full prefix (medicine title > section path), keeping medicine identity, boxed warnings
    and LiverTox case headings;
  - splits only at line/sentence boundaries; table rows stay whole and repeat caption + header;
  - last-resort split never separates a dose from its unit (flagged `forced_split`);
  - chunk score = max over windows; ties keep fused order; chunk ids and citation metadata unchanged;
  - raises instead of truncating when question + prefix leave too little room.
- Measured: 2.9 (B) / 6.1 (C) windowed chunks per question; 3 / 6 forced splits; no window over the limit.
- Tests: `tests/test_rerank.py`, 8 tests (fake scorer), all passing.

## 5. Retrieval quality (offline, `scripts/phase4/eval_rerank.py`)

| Metric | A baseline | B MiniLM, top 20 | C MiniLM, top 40 |
|---|---:|---:|---:|
| AP@8 (gold-based context-precision proxy) | 0.597 | **0.625** | 0.624 |
| MRR@8 | 0.747 | **0.814** | 0.812 |
| nDCG@8 | 0.665 | 0.692 | 0.688 |
| Evidence Recall@8 | 0.873 | 0.873 | 0.866 |
| Wrong-medicine passages in top 8 (per question) | 0.79 | 0.69 | 0.73 |

AP@8 is not Ragas Context Precision (judge-based; baseline 0.717). Arm A reproduces the Ragas run's Evidence
Recall@8 exactly (0.8731).

AP@8 by source, A → B: DailyMed 0.599 → 0.616; FDA 0.70 → 0.80; PMC 0.520 → 0.667; **LiverTox 0.688 → 0.396**.

AP@8 by category, A → B: ingredients 0.72 → 0.88; PMC 0.52 → 0.67; FDA 0.70 → 0.80; multi-document 0.18 → 0.24;
interactions 0.74 → 0.68; pharmacology 0.58 → 0.52; **LiverTox 0.69 → 0.40**.

## 6. Rank-distribution changes (first gold passage)

| Rank | A | B | C |
|---|---:|---:|---:|
| 1 | 44 | **50** | **50** |
| 2–3 | 12 | 8 | 8 |
| 4–8 | 5 | 4 | 3 |
| > 8 | 6 | 5 | 6 |

## 7. Multi-document failure analysis (8 questions)

| Question | Fused rank, evidence 1 / 2 | In B top 8 | Diagnosis |
|---|---|---|---|
| q071 Cozaar + Celebrex | 8 / 10 | ✓ / ✗ (C: ✗ / ✗) | Low ranking; C worse |
| q075 gabapentin label + LiverTox | 27 / 16 | ✗ / ✓ | Low ranking |
| q076 Lyrica + LiverTox | not in pool / 2 | ✗ / ✓ | Missing candidate |
| q077 Lipitor + LiverTox case | not in pool / 26 | ✗ / ✗ | Missing candidate + low ranking |
| q079 baclofen label + PMC | 35 / 1 | ✗ / ✓ | Low ranking |
| q080 gabapentin label + PMC | 43 / 35 | ✗ / ✗ | Low ranking; item 1 beyond top 40 |
| q083 rosuvastatin label + PMC | 4 / 10 | ✓ / ✗ | Low ranking |
| q085 buspirone label + PMC | 33 / 1 | ✗ / ✓ | Low ranking |

Pattern: one document dominates the top ranks; the second required source sits at ranks 10–43. Reranking within the
pool does not fix this. Likely remedies are retrieval-design changes (per-source or per-medicine diversity, one search
per named medicine) — not yet proposed for implementation. Gold-label narrowness not yet verified; gold labels are not
changed to raise scores. No conflicting clinical evidence involved.

## 8. Latency

| | B (top 20) | C (top 40) |
|---|---:|---:|
| Pairs per question (mean) | 22.9 | 46.2 |
| Windowing P50 / P95 | 0.13 / 0.28 s | 0.29 / 0.56 s |
| Inference P50 / P95 / P99 | 3.50 / 4.87 / 5.85 s | 7.82 / 9.69 / 12.79 s |
| **Rerank total P50 / P95 / P99** | **3.60 / 5.06 / 6.17 s** | **8.17 / 10.22 / 13.08 s** |
| Budget (experimental) | ≤ 0.5 s P95 | ≤ 0.5 s P95 |

- About 10× (B) and 20× (C) over budget. The static estimate (60–120 ms) was wrong by about 30×.
- Measured sequentially, single process, 4 CPU threads, nothing concurrent. Concurrency 1/10/25, CPU/memory profiling
  and cold vs warm (Step 6) not done; they can only add latency.
- The v2 dev Ragas latency (5.7 s / 9.5 s) is **not a valid end-to-end baseline**: it includes 2.8 s P50 / 5.2 s P95
  of thread-pool queuing behind judge calls. Retrieval alone: 0.18 s P50 / 0.56 s P95; LLM 2.7 s / 5.5 s.

## 9. Per-question regression analysis (B vs A)

16 improved, 9 regressed, 42 unchanged (AP@8). C adds q071 (lost an evidence item).

| Question | Category | ΔAP@8 | ΔEvidence recall | Observation |
|---|---|---:|---:|---|
| **q044** | interactions | −0.50 | **−1.00** | **Naproxen pregnancy section displaced the ibuprofen evidence** |
| q060 | pharmacology | −0.83 | 0 | Carcinogenesis section outranked the Zocor/telithromycin table |
| q050 | interactions | −0.75 | 0 | Ranking drop |
| **q116** | livertox | −0.67 | 0 | **Annotated bibliography outranked the clinical overview** |
| q037, q039 | interactions | −0.50 | 0 | Ranking drop |
| q112 | pmc | −0.50 | 0 | Ranking drop |
| **q113** | livertox | −0.50 | **−0.50** | **Lab-value table and bibliography displaced the case comment** |
| q117 | livertox | −0.25 | 0 | Ranking drop |

Largest improvements: q100 (+0.83), q003, q019, q035 (+0.67), q018, q105, q109, q135 (+0.50), q093, q029, q124;
small gains for multi-document q076, q083, q075.

Per-question deltas: `data/processed/eval/rerank/rr_B_minilm20/paired_vs_baseline.csv`,
`data/processed/eval/rerank/rr_C_minilm40/paired_vs_baseline.csv`.

## 10. Clinical evidence-integrity findings

1. **Wrong-drug ranking (q044):** NSAID class text in another drug's label outranked the asked drug. Overall
   wrong-medicine passages fell (0.79 → 0.69), but this single case is clinically significant.
2. **Bibliography bias:** LiverTox reference lists outrank clinical evidence (q113, q116); LiverTox AP 0.69 → 0.40.
3. **Windowing behaved as designed:** no truncation, prefix in every window, ids preserved, 3–6 forced splits.
4. The LLM was not run, so no unsupported-claim or citation findings exist for the reranker arms (Step 7).

## 11. Recommendation

| Acceptance criterion | MiniLM B / C |
|---|---|
| Meaningful, reproducible Context Precision gain | Partial (AP@8 +0.028); not judge-verified |
| Evidence Recall@8 improved or preserved | Preserved / slightly lower |
| No clinically significant ranking regression | **Fails** (q044 wrong drug; LiverTox) |
| Reranker P95 ≤ 500 ms | **Fails** (5.1 s / 10.2 s) |
| Reproducible configuration and model version | Met |

**Retain the existing hybrid retrieval baseline. Do not advance MiniLM to Step 7. No production reranker is
recommended.**

## 12. Remaining risks and uncertainties

- Small sample (67 answerable; 4–15 per category); no confidence intervals computed yet.
- AP@8 is a proxy; Ragas judge variance unmeasured.
- MedCPT untested; may avoid bibliography and wrong-drug failures but cannot meet the latency budget on this CPU.
- Latency measured sequentially only; Step 6 profile outstanding.
- Multi-document gold labels unverified for narrowness.
- Disclaimer effect on faithfulness (0.847 with vs 0.992 without) is separate from reranking and undecided.

## 13. Open decisions

1. MedCPT quality-only test (license check, ~440 MB download).
2. Stop reranking on current hardware.
3. Retrieval-design proposals for multi-document questions (per-source/per-medicine diversity; bibliography exclusion).
4. Disclaimer experiment (generation, kept separate).
5. Commit of the Phase 4 code (currently uncommitted by request).

## Artifacts

| Item | Path |
|---|---|
| Candidate extraction | `scripts/phase4/extract_candidates.py`; `data/processed/eval/candidates/` |
| Reranker | `src/medical_rag/search/rerank.py`; `tests/test_rerank.py` |
| Offline evaluator | `scripts/phase4/eval_rerank.py`; `data/processed/eval/rerank/rr_{A_baseline,B_minilm20,C_minilm40}/` |
| v2 dev Ragas run (baseline) | `data/processed/eval/reports/v2_dev75_run1.json` (+ rows, judge trace) |
