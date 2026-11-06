# Demo tracker (10-hour plan)

Start 2026-11-04 17:47 IST · Deadline 2026-11-05 03:47 IST · Hardware: laptop CPU (i5-1155G7, 4 cores), no GPU.
Quality is not traded for time: retrieval metrics and Ragas evaluation stay in scope. Demo defaults are provisional (docs/DECISION_LOG.md ADR-011–017).

| Hour (IST) | Block | Status | Evidence |
|---|---|---|---|
| H1 17:47–18:47 | Install pinned embedding stack; download bge-base-en-v1.5; tokenizer + speed check | Done | Weights sha256 OK; vocab identical, 0/5,815 chunks differ; 2.4 chunks/s (batch 1, 4 threads) |
| H2 18:47–19:47 | Embedding module + tests; embed 5,815 chunks (overlap 64) | Done 19:46 | 5,815 vectors in 77.4 min (1.25 chunks/s with other work on the CPU); 91 inputs over 512 tokens truncated at the end (listed in manifest) |
| H3 19:47–20:47 | OpenSearch index: HNSW vectors + BM25 text; integration tests | Done 19:48 | Index medrag-chunks-f6fab776e222268a-1bbc080536298982, 5,815 docs, 17.3 s; alias medrag-chunks; 4-query smoke check: hybrid ~200 ms warm (query embedding ~170 ms of it) |
| H4 20:47–21:47 | Hybrid retrieval (BM25 + dense, RRF); /search API | Done (ahead) | 8 unit + 4 API tests pass; server at http://localhost:8000/docs. Fix (user-approved): one pooled OpenSearch client + 127.0.0.1 → /search total 149–258 ms over 8 queries (was ~2,150 ms) |
| H5 21:47–22:47 | Retrieval test set (~150 questions) drafted → user review; recall@k, MRR, nDCG | Scripts done; **blocked on API key** | draft_testset.py, eval_retrieval.py; metrics unit-tested |
| H6–7 22:47–00:47 | Grounded answers with citations and abstention (Claude API); /ask API | Code done (ahead) | 6 unit tests with fake LLM; live call untested until key |
| H8 00:47–01:47 | Ragas evaluation (faithfulness, answer relevancy, context precision/recall, factual correctness) | Script done | Ragas 0.4.3; judge via own Claude adapter (Ragas's Anthropic adapter forces tool use, rejected by current models) |
| H9 01:47–02:47 | Latency P50/P95, throughput, component timings | Script done | bench_latency.py |
| H10 02:47–03:47 | Demo script, README/docs, commit (with approval), buffer | Pending | |

## UI (user request 20:00)
Single page at http://localhost:8000/ served by FastAPI (no build step, no external assets): compare BM25 / dense /
hybrid side by side, source filters, query-term highlighting, chunk inspector (metadata, prefix, previous/next chunk,
overlap shading), relevance judgments kept in the browser with JSON export, Ask tab (disabled until the API key is
set). New endpoints: GET /info, GET /chunks/{id}. Index rebuilt with notices, split_levels, overlap_tokens (vectors
unchanged). Embedder loads at startup (~35 s); first 3-retriever search 748 ms (was 23.6 s cold).

## Decisions taken for the demo (2026-11-04, user)
Model bge-base-en-v1.5; overlap 64; HNSW (Lucene); BM25; RRF; no reranker; Claude API for answers and Ragas judge; Swagger UI; transformers 5.22.1.

## Package notes
- transformers 5.22.1 (user choice); ragas 0.4.3 needed langchain-community 0.3.31 (0.4.2 fails to import in Ragas).
- Generator and judge model default `claude-opus-5-5` (MEDRAG_LLM_MODEL, effort medium); judge can be changed with `--judge-model`.

## Key and live checks (20:55)
- Key was placed in the tracked `.env.example`; moved to gitignored `.env` without printing it, placeholder restored,
  confirmed absent from Git history.
- Live /ask: gabapentin renal dosing answered with 6 valid citations (8.1 s LLM, 4,082 input tokens); semaglutide
  (not in corpus) correctly abstained (2.0 s).
- Ragas live check: all 5 metrics run through the Claude judge adapter (11 judge calls for one sample).

## Test set and retrieval evaluation (21:07, draft test set, pre-review)
- retrieval_v1: 145 questions (DailyMed 80, PMC 25, LiverTox 15, FDA 10 — 9 FDA chunks unsuitable — and 15
  unanswerable); every gold quote verified in its chunk; 125 questions have 1 gold chunk, 5 have 2 (overlap copies).
- 130 answerable questions, k = 10 (report data/processed/eval/reports/retrieval_retrieval_v1.json):

| Run | hit@5 | recall@10 | MRR@10 | nDCG@10 | p50 ms |
|---|---|---|---|---|---|
| BM25 | 0.677 | 0.800 | 0.525 | 0.591 | 16 |
| Dense + instruction | **0.923** | 0.946 | **0.783** | **0.821** | 181 |
| Hybrid RRF + instruction | 0.885 | **0.954** | 0.706 | 0.765 | 189 |
| Dense, no instruction | 0.900 | 0.954 | 0.776 | 0.816 | 165 |
| Hybrid RRF, no instruction | 0.885 | 0.931 | 0.700 | 0.755 | 190 |

- Caveat: questions were written by Claude from the gold chunk, which can favour dense retrieval (paraphrase).

## Beyond Doc chat UI (user request 00:00, plan approved 00:15)
- React 19 + Vite 8 app (`frontend/`) per the Claude Design handoff, adapted to the project corpus; served at `/`
  (explorer moved to `/explorer`). Home → chat with live pipeline stages (SSE), citation chips, evidence panel
  (Sources / Network / Notes), follow-ups, chat history per anonymous user, settings (theme, k, mode, sources).
- Backend: `src/medical_rag/chat/` (ClickHouse history, Redis memory + rate limit, turn pipeline with follow-up
  rewrite), `api/chat.py` (sessions CRUD + streaming messages). 273 unit + 9 integration tests pass.
- Browser check (01:05): gabapentin renal dosing → 5 cited passages, 6.4 s; follow-up "And what about elderly
  patients?" rewritten to a gabapentin query (3.2 s rewrite) → 3 cited, 6.2 s; history reloads from ClickHouse.
- Known: the follow-up rewrite adds ~2–3 s (one extra LLM call); network graph links medicines only (no concept
  extraction); notes are browser-only.

## Ragas run 1 (2026-11-05, unreviewed draft test set, 145 questions)
Generator claude-sonnet-5-5 (effort medium, k=8, hybrid); judge Kimi K3 on Fireworks
(`accounts/fireworks/models/kimi-k3`, reasoning low, temperature 0) — user decision. Report:
`data/processed/eval/reports/ragas_retrieval_v1_145q.json` (+ `_rows.jsonl`).

| Metric (128 answered) | Mean |
|---|---|
| Faithfulness | 0.870 |
| Answer relevancy | 0.859 |
| Context precision | 0.791 |
| Context recall | 0.993 |
| Factual recall (reference facts in answer) | 0.800 |
| Factual precision (answer claims in 1-sentence reference) | 0.410 |
| Factual correctness F1 | 0.489 |

- Abstention: 13/15 unanswerable abstained; 2/130 answerable falsely abstained (both: gold chunk not retrieved).
  The 2 "unanswerable" answers (q139 tacrolimus, q145 warfarin phone) were grounded in corpus passages, so those two
  test items are mislabelled rather than model errors.
- Citations: 0 invalid ids; gold chunk retrieved 94.6%, cited when answered 95.3%.
- Latency (concurrency 4): end-to-end P50 10.8 s / P95 15.4 s; Sonnet P50 2.8 s / P95 5.5 s.
- Judge: 2,560 calls, 3.57 M input (0.28 M cached) + 0.76 M output tokens, ≈ $21.41 (Fireworks list prices).
- Pilot finding: factual-correctness F1 is dominated by precision against short references (extra correct detail
  is penalised); reported split. Faithfulness covers support by the retrieved passages.

## Models (user decision 21:10)
- Answer generation: **claude-sonnet-5-5** (MEDRAG_LLM_MODEL), effort medium. Ragas judge: **claude-opus-5-5**
  (MEDRAG_JUDGE_MODEL), so the generator does not grade itself.
- Sonnet 5.5 live check: gabapentin renal dosing answered with valid citations in 4.0 s (Opus 5.5: 8.1 s);
  semaglutide abstained in 2.1 s.

## Waiting on the user
- Review of the drafted test set (data/eval/retrieval_v1_review.csv); Ragas starts after the review (user, 21:10).
