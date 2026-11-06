# Retrieval v1.2 — medicine-, source- and section-aware hybrid search

Status: **experimental preset `v12_t3`, selected on the v2.0 dev split and checked once on the holdout split.**
Not a final architectural decision (ADR-022 in docs/DECISION_LOG.md). Off by default (`MEDRAG_RETRIEVAL_PLAN=none`).
Informational demo only; not medical advice.

## What stayed fixed
Corpus (100 documents, 5,815 chunks), chunking, embedding model (bge-base-en-v1.5), OpenSearch index and mapping,
BM25 + dense with client-side RRF (k = 60), top k = 8, 50 candidates per retriever, generator (Claude Sonnet 5.5),
judge (Kimi K3 on Fireworks), and the frozen v2.0 dataset (75 dev / 75 holdout). No reranker, no agentic retrieval,
no LLM in the planner, no reindexing.

## What v1.2 adds (all deterministic, corpus-only, swappable)
| Step | Component | Module |
|---|---|---|
| 2 | Medicine resolver: alias table built from the corpus (brand, ingredient, salt forms, LiverTox trade names); confidence verified / derived / ambiguous / none | `query/medicine.py`, `scripts/phase5/build_medicine_aliases.py` |
| 3 | Planner: soft BM25 medicine boost + a medicine-filtered dense list in RRF (verified only); optional identity guard (not used in T3) | `search/planner.py`, `search/retrieval.py` |
| 4 | Coverage: one filtered BM25 + k-NN branch per (medicine, source) when 2+ sources are named, or per medicine for 2+ medicines; run concurrently (0.3 s timeout, failures skipped); coverage floor 2 per unit, capped at k/2 | `search/planner.py`, `search/selection.py` |
| 5 | Single-study rule (no per-medicine units for a one-study question with < 2 named sources); noise demotion (LiverTox bibliography, PMC back matter, FDA letter signatures, DailyMed patient sections; at most 1 kept in place, never deleted; off for reference questions); intent → section soft boost (fixed table, boost 1.0) | `query/intent.py`, `search/planner.py`, `search/selection.py` |

With no plan (`MEDRAG_RETRIEVAL_PLAN=none`) search is byte-identical to the pre-v1.2 baseline (verified on all 75 dev
questions against saved candidates). With T3, noise demotion applies to every question except reference questions,
so most questions take the plan path.

## Offline retrieval results (top 8; answerable questions)
Evidence Recall@8 = share of gold evidence items with an acceptable chunk in the top 8. AP@8 is the gold-based
stand-in for context precision; `AP dedup` counts identical evidence sets once (about 15 questions have two evidence
items pointing at the same chunk, which capped their AP at 0.5 — a metric artefact, gold labels unchanged).
Wrong-medicine = passages in the top 8 from documents of a medicine not in the gold set.

### Dev (67 answerable) — selection split
| Arm | ER@8 | AP@8 | AP dedup | nDCG@8 | MRR@8 | Wrong-med | Evidence docs present |
|---|---|---|---|---|---|---|---|
| A0 baseline hybrid | 0.873 | 0.597 | 0.703 | 0.665 | 0.747 | 1.22 | 0.985 |
| M2 medicine boost + filtered dense | 0.896 | 0.617 | 0.731 | 0.686 | 0.778 | 0.13 | 0.970 |
| S2 + coverage floor 2 | 0.910 | 0.621 | 0.735 | 0.692 | 0.778 | 0.19 | 0.993 |
| SC + single-study rule | 0.910 | 0.621 | 0.735 | 0.692 | 0.778 | 0.13 | 0.993 |
| T1 SC + noise demotion | 0.910 | 0.626 | 0.740 | 0.696 | 0.784 | 0.12 | 0.993 |
| T2 SC + section boost 1.0 | 0.925 | 0.647 | 0.766 | 0.715 | 0.804 | 0.16 | 1.000 |
| **T3 SC + noise + section boost 1.0** | **0.925** | **0.650** | **0.769** | **0.718** | **0.809** | 0.15 | **1.000** |
| T4 T3 with section boost 2.0 | 0.925 | 0.624 | 0.738 | 0.697 | 0.778 | 0.24 | 1.000 |

T3 against A0 on dev: recall improved on 5 questions, regressed on 0; AP improved on 12, regressed on 4.
T3 against S2: AP improved on 9 questions, regressed on 2 (q060, q083), recall improved on 2 (q079, q085), regressed 0.
Multi-document category ER@8: A0 0.31 → T3 0.63.

### Holdout (68 answerable) — run once, after selection
| Arm | ER@8 | AP@8 | AP dedup | nDCG@8 | MRR@8 | Wrong-med | Evidence docs present |
|---|---|---|---|---|---|---|---|
| A0 | 0.904 | 0.626 | 0.715 | 0.697 | 0.771 | 0.87 | 0.978 |
| M2 | 0.919 | 0.623 | 0.710 | 0.697 | 0.760 | 0.15 | 0.978 |
| S2 | 0.934 | 0.629 | 0.716 | 0.705 | 0.765 | 0.24 | 1.000 |
| **T3** | **0.949** | **0.650** | **0.737** | **0.724** | **0.786** | 0.19 | **1.000** |

T3 against A0 on holdout: recall improved on 5, regressed on 0; AP improved on 13, regressed on 5.

## End-to-end Ragas (dev 75; same generator, judge and settings; one judge run each)
| Metric | Baseline `v2_dev75_run1` | T3 `v2_dev75_t3_run1` | Change |
|---|---|---|---|
| Faithfulness | 0.914 | 0.927 | +0.013 |
| Answer relevancy | 0.873 | 0.868 | −0.005 |
| Context precision | 0.717 | 0.766 | +0.049 |
| Context recall | 0.934 | 0.966 | +0.032 |
| Claim recall | 0.738 | 0.772 | +0.035 |
| Factual precision | 0.442 | 0.453 | +0.011 |
| Factual F1 | 0.478 | 0.504 | +0.026 |
| Gold passage retrieved (top 8) | 0.910 | 0.955 | +0.045 |
| Gold passage cited when answered | 0.896 | 0.940 | +0.045 |
| Unanswerable correctly abstained | 8 / 8 | 8 / 8 | — |
| False abstentions (answerable) | 0 / 67 | 0 / 67 | — |
| Invalid citation ids | 0 | 0 | — |
| End-to-end P50 / P95 (concurrency 4) | 5.71 s / 9.54 s | 5.65 s / 9.49 s | ≈ same |

Single judge run per arm: differences under about 0.01–0.02 (answer relevancy, factual precision) are within the judge
variance seen earlier and should not be read as real changes. Judge cost about USD 12.8 for the T3 run.

## Retrieval latency benchmark (`bench_run1`)
All 150 frozen questions (text only), sequential, one warm-up pass, 3 measured passes, arm order alternated; single
local OpenSearch node, laptop CPU (i5-1155G7). Retrieval only, no LLM.

| ms | A0 P50 | A0 P95 | A0 P99 | T3 P50 | T3 P95 | T3 P99 |
|---|---|---|---|---|---|---|
| Total | 171 | 291 | 578 | 191 | 323 | 550 |
| Query embedding | 144 | 250 | 525 | 145 | 249 | 348 |
| BM25 | 12 | 24 | 39 | 13 | 28 | 45 |
| Dense (broad) | 14 | 29 | 41 | 14 | 26 | 48 |
| Dense (medicine-filtered; 381 of 450 queries) | — | — | — | 13 | 25 | 42 |
| Coverage branches (75 of 450 queries) | — | — | — | 29 | 95 | 153 |

T3 adds about 20 ms at P50 and 30 ms at P95. Query embedding on CPU dominates both arms. The coverage branches stayed
well under their 0.3 s timeout (P99 153 ms). Generation (Claude, P50 about 2.6 s) dominates end-to-end latency.

## Known limitations
- Rules (intent lexicon, section table, noise list) were designed by inspecting dev failures; the holdout run is the
  only unbiased check, and it is small (68 answerable questions).
- q076 and q079-style questions with implied sources, and in-document passage choice (q077, q080, q083) remain partly
  unsolved.
- Medicine names per document come from document-level metadata (1–2 names); PMC papers comparing two drugs list one.
- Latency numbers are from a single local node on a laptop CPU, sequential; not an SLO.

## Reproduce
```
.venv/Scripts/python -X utf8 scripts/phase5/build_medicine_aliases.py
.venv/Scripts/python -X utf8 scripts/phase5/eval_v12.py --out-name <name> --arms A0,M2,S2,SC,T1,T2,T3,T4
.venv/Scripts/python -X utf8 scripts/phase5/eval_v12.py --out-name <name> --split holdout --arms A0,M2,S2,T3
.venv/Scripts/python -X utf8 scripts/phase3/eval_ragas.py --testset data/eval/v2/frozen/retrieval_v2.0_split75-75.jsonl --split dev --plan v12_t3 --out-name <name>
.venv/Scripts/python -X utf8 scripts/phase5/bench_retrieval.py --out-name <name>
```
