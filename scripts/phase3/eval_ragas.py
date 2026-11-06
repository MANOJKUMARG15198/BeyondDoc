"""End-to-end RAG evaluation with Ragas: hybrid retrieval -> grounded answer -> Ragas metrics + abstention checks.

Ragas metrics (judge: MEDRAG_JUDGE_PROVIDER — Kimi K3 on Fireworks AI by default, or Claude — through the project's
structured-output adapter; embeddings: pinned bge):
faithfulness, answer_relevancy, context_precision, context_recall, factual_precision (answer claims supported by the
reference). Reference-claim recall uses the project's ClaimRecall (Ragas prompts, one direction only); Ragas's own
FactualCorrectness(mode="recall") is kept as factual_recall_legacy because it mixes in the precision direction.
Non-LLM checks: abstention, false abstention, citation validity, per-sentence citation coverage, gold chunk rank
(top-k and full candidate pool, strict gold and with human-approved equivalents), duplicate passages, latency.

Each run writes a manifest (git commit, test-set hash, concrete index, package versions, settings). Existing reports
are never overwritten unless --overwrite is given, so the frozen Run 1 baseline stays intact.

v2 test sets (detected from the file) add Evidence Recall@K over each item's evidence sets, per-category, per-split
and per-difficulty breakdowns, and abstention checks for unanswerable and insufficient-evidence items; --split selects
dev (default), holdout or all, so the holdout is only scored when asked for explicitly.

.venv/Scripts/python -X utf8 scripts/phase3/eval_ragas.py --testset data/eval/retrieval_v1.1.jsonl --out-name run2
.venv/Scripts/python -X utf8 scripts/phase3/eval_ragas.py --testset data/eval/v2/frozen/retrieval_v2.0.jsonl \
    --split dev --out-name v2_dev_run1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from medical_rag.core.config import get_settings  # noqa: E402
from medical_rag.embedding.config import BGE_QUERY_INSTRUCTION, EmbeddingConfig  # noqa: E402
from medical_rag.embedding.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from medical_rag.evaluation.citations import citation_coverage  # noqa: E402
from medical_rag.evaluation.claim_recall import ClaimRecall  # noqa: E402
from medical_rag.evaluation.manifest import run_manifest  # noqa: E402
from medical_rag.evaluation.ragas_adapters import JUDGE_TAG, BgeRagasEmbedding, ClaudeJudge  # noqa: E402
from medical_rag.evaluation.testset import TestItem, load_testset, squash  # noqa: E402
from medical_rag.evaluation.testset_v2 import evidence_recall, is_v2_file, load_v2, to_eval_item  # noqa: E402
from medical_rag.generation.answerer import answer_question  # noqa: E402
from medical_rag.generation.claude import ClaudeLLM  # noqa: E402
from medical_rag.query.medicine import AliasTable  # noqa: E402
from medical_rag.search.client import make_client  # noqa: E402
from medical_rag.search.planner import plan_question  # noqa: E402
from medical_rag.search.retrieval import Retriever  # noqa: E402

REPORTS = Path("data/processed/eval/reports")
SCORED = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_precision",
    "claim_recall",
    "factual_recall_legacy",
    "factual_f1",
)


def pct(values: list[float], q: int) -> float | None:
    if not values:
        return None
    return round(statistics.quantiles(values, n=100)[q - 1], 1) if len(values) > 1 else round(values[0], 1)


def _judge_cost(llm) -> float | None:
    usage = getattr(llm, "usage", None)
    if not usage:
        return None
    from medical_rag.generation.openai_compat import cost_usd

    return cost_usd(usage)


def pick(items: list[TestItem], limit: int | None) -> list[TestItem]:
    """Deterministic stratified subset: keep the per-source mix, always include every unanswerable question."""
    answerable = [i for i in items if i.kind == "answerable"]
    unanswerable = [i for i in items if i.kind == "unanswerable"]
    if limit is None or limit >= len(answerable):
        return answerable + unanswerable
    by_source: dict[str, list[TestItem]] = defaultdict(list)
    for i in answerable:
        by_source[i.source or ""].append(i)
    chosen: list[TestItem] = []
    for _source, group in sorted(by_source.items()):
        share = max(1, round(limit * len(group) / len(answerable)))
        chosen += group[:share]
    return chosen[:limit] + unanswerable


def first_rank(ranked: list[str], gold: set[str]) -> int | None:
    return next((i for i, c in enumerate(ranked, start=1) if c in gold), None)


def f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


async def score(metrics: dict, row: dict, tag: str) -> dict:
    """All metrics for one answer run concurrently; the judge's own semaphore caps total in-flight calls."""
    q, a, ctx, ref = row["question"], row["answer"], row["contexts"], row["reference_answer"]
    calls = {
        "faithfulness": lambda: metrics["faithfulness"].ascore(user_input=q, response=a, retrieved_contexts=ctx),
        "answer_relevancy": lambda: metrics["answer_relevancy"].ascore(user_input=q, response=a),
        "context_precision": lambda: metrics["context_precision"].ascore(
            user_input=q, reference=ref, retrieved_contexts=ctx
        ),
        "context_recall": lambda: metrics["context_recall"].ascore(user_input=q, retrieved_contexts=ctx, reference=ref),
        # Precision: answer claims supported by the short reference (penalises extra correct detail).
        "factual_precision": lambda: metrics["factual_precision"].ascore(response=a, reference=ref),
        # Recall: reference claims stated in the answer (one-direction metric) and Ragas's mixed legacy value.
        "claim_recall": lambda: metrics["claim_recall"].ascore(response=a, reference=ref),
        "factual_recall_legacy": lambda: metrics["factual_recall_legacy"].ascore(response=a, reference=ref),
    }

    async def one(name: str, call) -> tuple[str, float | None, str | None, dict | None]:
        JUDGE_TAG.set(f"{tag}:{name}")  # this task's own context copy
        try:
            result = await call()
            detail = result.as_dict() if hasattr(result, "as_dict") else None
            return name, None if result.value is None else float(result.value), None, detail
        except Exception as exc:  # noqa: BLE001 - a failed metric is recorded, not fatal
            return name, None, f"{type(exc).__name__}: {exc}"[:300], None

    out: dict = {}
    for name, value, err, detail in await asyncio.gather(*(one(n, c) for n, c in calls.items())):
        out[name] = value
        if err:
            out.setdefault("errors", {})[name] = err
        if detail:
            out[f"{name}_detail"] = detail
    out["factual_f1"] = f1(out.get("factual_precision"), out.get("claim_recall"))
    return out


def combine(runs: list[dict]) -> dict:
    """Mean of each metric over repeated judge runs; per-metric spread kept for the variance report."""
    out: dict = {}
    for name in SCORED:
        values = [r[name] for r in runs if r.get(name) is not None]
        out[name] = round(statistics.fmean(values), 4) if values else None
        if len(runs) > 1:
            out.setdefault("spread", {})[name] = round(max(values) - min(values), 4) if values else None
    for key in ("errors", "claim_recall_detail"):
        if key in runs[0]:
            out[key] = runs[0][key]
    return out


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def breakdown(rows: list[dict], key: str, k: int) -> dict:
    """Per-group (category, split or difficulty) Ragas means, Evidence Recall@K and abstention counts."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key))].append(r)
    out = {}
    for name, group in sorted(groups.items()):
        answerable = [r for r in group if r["kind"] == "answerable"]
        other = [r for r in group if r["kind"] != "answerable"]
        scored = [r["ragas"] for r in group if "ragas" in r]
        out[name] = {
            "questions": len(group),
            "answerable": len(answerable),
            "false_abstentions": sum(r["abstained"] for r in answerable),
            "not_answerable_abstained": f"{sum(r['abstained'] for r in other)}/{len(other)}",
            f"evidence_recall@{k}": _mean([r.get("evidence_recall_top_k") for r in answerable]),
            "ragas": {m: _mean([s.get(m) for s in scored]) for m in SCORED},
        }
    return out


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    name = args.out_name or f"ragas_{args.testset.stem}_{args.limit or 'all'}"
    outputs = [REPORTS / f"{name}{suffix}" for suffix in (".json", "_rows.jsonl", "_judge_trace.jsonl")]
    if not args.overwrite and any(p.exists() for p in outputs):
        print(f"refusing to overwrite existing report {name}; pass --out-name or --overwrite", file=sys.stderr)
        return 2

    v2 = is_v2_file(args.testset)
    if v2:  # frozen v2 sets: every item is reviewed; select the split explicitly
        loaded = [to_eval_item(i) for i in load_v2(args.testset) if i.review.medical_review != "rejected"]
        loaded = [i for i in loaded if args.split == "all" or i.split == args.split]
    else:
        loaded = [i for i in load_testset(args.testset) if i.review_status != "rejected"]
    items = pick(loaded, args.limit)
    if args.answerable_only:
        items = [i for i in items if i.kind == "answerable"]
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [i for i in items if i.id in wanted]
    embedder = SentenceTransformerEmbedder(EmbeddingConfig(query_instruction=BGE_QUERY_INSTRUCTION))
    client = make_client(settings)
    retriever = Retriever(client, embedder, index=settings.index_alias)
    llm = ClaudeLLM(settings)
    judge_model = args.judge_model or settings.judge_model
    if settings.judge_provider == "fireworks":
        from medical_rag.generation.openai_compat import OpenAICompatLLM

        judge_settings = settings.model_copy(update={"judge_model": judge_model})
        judge_llm = OpenAICompatLLM.fireworks_judge(judge_settings)  # the adapter works with any complete_json LLM
    else:
        judge_llm = ClaudeLLM(settings.model_copy(update={"llm_model": judge_model}))
    judge = ClaudeJudge(judge_llm, max_concurrency=args.judge_concurrency, record=args.record_judge)
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge, embeddings=BgeRagasEmbedding(embedder.with_query_instruction(None))
        ),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
        "factual_precision": FactualCorrectness(llm=judge, mode="precision"),
        "claim_recall": ClaimRecall(llm=judge),
        "factual_recall_legacy": FactualCorrectness(llm=judge, mode="recall"),
    }
    manifest = run_manifest(
        testset=args.testset,
        client=client,
        alias=settings.index_alias,
        config={
            "generator": llm.model,
            "generator_effort": settings.llm_effort,
            "judge_provider": settings.judge_provider,
            "judge": judge_model,
            "judge_reasoning_effort": settings.judge_reasoning_effort,
            "judge_temperature": settings.judge_temperature,
            "judge_repeats": args.judge_repeats,
            "embedder": embedder.config.embedder_id,
            "query_instruction": embedder.config.query_instruction,
            "mode": "hybrid",
            "k": args.k,
            "retrieval_plan": args.plan,
            "answer_prompt": args.prompt,
            "candidates": args.candidates,
            "concurrency": args.concurrency,
            "testset_format": "v2" if v2 else "v1",
            "split": args.split if v2 else None,
            "questions": [i.id for i in items],
        },
    )
    print(f"{len(items)} questions; generator {llm.model}, judge {judge.llm.model} x{args.judge_repeats}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    table = None
    if args.plan == "v12_t3":  # Retrieval v1.2 preset; the plan's selection applies to the final k, so k = args.k
        alias_path = args.aliases or sorted(Path("data/processed/medicine").glob("aliases_*.json"))[0]
        table = AliasTable.load(Path(alias_path))

    async def run_one(item: TestItem) -> dict:
        async with sem:
            t0 = time.perf_counter()
            # The whole fused candidate list (up to 2 x candidates) is requested so the gold chunk's position in the
            # pool can be recorded; the answer still gets only the top k, identical to search(k=k) for hybrid mode.
            if table is None:
                result = await asyncio.to_thread(
                    retriever.search, item.question, k=2 * args.candidates, candidates=args.candidates
                )
            else:
                plan = plan_question(item.question, table)
                result = await asyncio.to_thread(
                    retriever.search, item.question, k=args.k, candidates=args.candidates, plan=plan
                )
            hits = result.hits[: args.k]
            answer = await asyncio.to_thread(
                answer_question, llm, item.question, [h.source for h in hits], (), args.prompt
            )
            e2e = (time.perf_counter() - t0) * 1000
        pool = [h.chunk_id for h in result.hits]
        top = pool[: args.k]
        gold, gold_eq = set(item.gold_chunk_ids), item.gold_with_equivalents()
        cited = {c.chunk_id for c in answer.citations}
        bodies = [squash(h.source.get("text", "")) for h in hits]
        coverage = citation_coverage(answer.answer) if not answer.abstained else None
        row = {
            "id": item.id,
            "kind": item.kind,
            "source": item.source,
            "question": item.question,
            "reference_answer": item.reference_answer,
            "answer": answer.answer,
            "abstained": answer.abstained,
            "abstain_reason": answer.abstain_reason,
            "coverage": answer.llm_meta.get("coverage"),
            "contexts": answer.contexts,
            "retrieved": [
                {
                    "rank": i,
                    "chunk_id": h.chunk_id,
                    "document_id": h.source.get("document_id"),
                    "section_path": h.source.get("section_path"),
                    "score": round(h.score, 6),
                    "component_ranks": h.ranks,
                }
                for i, h in enumerate(hits, start=1)
            ],
            "pool_size": len(pool),
            "gold_rank_top_k": first_rank(top, gold),
            "gold_rank_in_pool": first_rank(pool, gold),
            "gold_rank_in_pool_equiv": first_rank(pool, gold_eq),
            "duplicate_passages": len(bodies) - len(set(bodies)),
            "cited_chunk_ids": [c.chunk_id for c in answer.citations],
            "invalid_citations": answer.llm_meta.get("invalid_citations", []),
            "citation_coverage": coverage.as_dict() if coverage else None,
            "gold_retrieved": bool(gold & set(top)),
            "gold_retrieved_equiv": bool(gold_eq & set(top)),
            "gold_cited": bool(gold & cited),
            "gold_cited_equiv": bool(gold_eq & cited),
            "timings_ms": {**result.timings_ms, **answer.timings_ms, "end_to_end_ms": round(e2e, 1)},
        }
        if v2:
            sets = item.evidence_sets
            row |= {
                "category": item.category,
                "split": item.split,
                "difficulty": item.difficulty,
                "answerability": item.answerability,
                "evidence_items": len(sets),
                "evidence_recall_top_k": evidence_recall(top, sets),
                "evidence_recall_pool": evidence_recall(pool, sets),
            }
        if item.kind == "answerable" and not answer.abstained:
            runs = [await score(metrics, row, f"{item.id}:run{n}") for n in range(args.judge_repeats)]
            row["ragas"] = combine(runs)
            if args.judge_repeats > 1:
                row["ragas_runs"] = [{k: r.get(k) for k in SCORED} for r in runs]
        print(f"  {item.id} {'abstained' if answer.abstained else 'answered'} {e2e / 1000:.1f}s", flush=True)
        return row

    rows = await asyncio.gather(*(run_one(i) for i in items))

    answerable = [r for r in rows if r["kind"] == "answerable"]
    unanswerable = [r for r in rows if r["kind"] == "unanswerable"]
    scored = [r["ragas"] for r in rows if "ragas" in r]
    means = {}
    for metric in SCORED:
        values = [s[metric] for s in scored if s.get(metric) is not None]
        means[metric] = {"mean": round(statistics.fmean(values), 4) if values else None, "n": len(values)}
    variance = None
    if args.judge_repeats > 1:
        variance = {}
        for metric in SCORED:
            spreads = [s["spread"][metric] for s in scored if s.get("spread", {}).get(metric) is not None]
            variance[metric] = {
                "mean_spread": round(statistics.fmean(spreads), 4) if spreads else None,
                "max_spread": max(spreads) if spreads else None,
                "questions_with_spread_over_0.2": sum(x > 0.2 for x in spreads),
            }
    coverages = [r["citation_coverage"] for r in rows if r.get("citation_coverage")]
    e2e = [r["timings_ms"]["end_to_end_ms"] for r in rows]
    llm_ms = [r["timings_ms"]["llm_ms"] for r in rows if "llm_ms" in r["timings_ms"]]

    def rate(values: list[bool]) -> float | None:
        return round(statistics.fmean(values), 4) if values else None

    report = {
        "manifest": manifest,
        "testset": str(args.testset),
        "generator": llm.model,
        "judge": judge.llm.model,
        "effort": settings.llm_effort,
        "k": args.k,
        "questions": len(rows),
        "ragas": means,
        "judge_variance": variance,
        "abstention": {
            "unanswerable_correctly_abstained": sum(r["abstained"] for r in unanswerable),
            "unanswerable_total": len(unanswerable),
            "answerable_false_abstentions": sum(r["abstained"] for r in answerable),
            "answerable_total": len(answerable),
        },
        "retrieval": {
            "gold_rank_top_k": dict(Counter(str(r["gold_rank_top_k"]) for r in answerable)),
            "gold_in_pool_rate": rate([r["gold_rank_in_pool"] is not None for r in answerable]),
            "gold_in_pool_rate_equiv": rate([r["gold_rank_in_pool_equiv"] is not None for r in answerable]),
            "gold_at_rank_1_rate": rate([r["gold_rank_top_k"] == 1 for r in answerable]),
            "questions_with_duplicate_passages": sum(r["duplicate_passages"] > 0 for r in rows),
        },
        "citations": {
            "answers_with_invalid_citation_ids": sum(bool(r["invalid_citations"]) for r in rows),
            "gold_retrieved_rate": rate([r["gold_retrieved"] for r in answerable]),
            "gold_retrieved_rate_equiv": rate([r["gold_retrieved_equiv"] for r in answerable]),
            "gold_cited_rate_when_answered": rate([r["gold_cited"] for r in answerable if not r["abstained"]]),
            "gold_cited_rate_when_answered_equiv": rate(
                [r["gold_cited_equiv"] for r in answerable if not r["abstained"]]
            ),
            "sentence_coverage_mean": rate([c["coverage"] for c in coverages if c["coverage"] is not None]),
            "answers_with_uncited_factual_sentences": sum(bool(c["uncited_factual"]) for c in coverages),
            "disclaimer_sentences": sum(len(c["disclaimers"]) for c in coverages),
        },
        "latency_ms": {
            "end_to_end": {"p50": pct(e2e, 50), "p95": pct(e2e, 95)},
            "llm": {"p50": pct(llm_ms, 50), "p95": pct(llm_ms, 95)},
            "note": f"measured with concurrency {args.concurrency} on a 4-core laptop CPU; not a throughput test",
        },
        "judge_calls": judge.calls,
        "judge_provider": settings.judge_provider,
        "judge_reasoning_effort": settings.judge_reasoning_effort if settings.judge_provider == "fireworks" else None,
        "judge_usage": getattr(judge.llm, "usage", None),
        "judge_cost_usd_estimate": _judge_cost(judge.llm),
        "testset_status": "unreviewed draft" if any(i.review_status == "draft" for i in items) else "reviewed",
    }
    if v2:
        report["evidence_recall"] = {
            f"recall@{args.k}": rate([r["evidence_recall_top_k"] for r in answerable]),
            "recall_in_pool": rate([r["evidence_recall_pool"] for r in answerable]),
            "all_evidence_in_top_k_rate": rate([r["evidence_recall_top_k"] == 1.0 for r in answerable]),
        }
        report["insufficient_evidence_review"] = {
            "rule": "correct if the system abstains, or (human/judge review) states the detail is not reported "
            "without giving any value; answered items are listed for that review",
            "abstained": sum(r["abstained"] for r in rows if r["answerability"] == "insufficient_evidence"),
            "total": sum(r["answerability"] == "insufficient_evidence" for r in rows),
            "answered_ids_for_review": [
                r["id"] for r in rows if r["answerability"] == "insufficient_evidence" and not r["abstained"]
            ],
        }
        report["breakdown"] = {key: breakdown(rows, key, args.k) for key in ("category", "split", "difficulty")}
    REPORTS.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, indent=2), encoding="utf-8")
    outputs[1].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if args.record_judge:
        outputs[2].write_text("".join(json.dumps(t) + "\n" for t in judge.trace), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ragas", "judge_variance", "retrieval", "citations")}, indent=2))
    print("wrote", *[str(p) for p in outputs if p.exists()])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="answerable questions (stratified); all unanswerable")
    parser.add_argument(
        "--split", choices=["dev", "holdout", "all"], default="dev", help="v2 test sets only; holdout must be explicit"
    )
    parser.add_argument("--ids", default=None, help="comma-separated question ids (e.g. a fixed variance subset)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=50, help="per-retriever candidates before fusion")
    parser.add_argument("--concurrency", type=int, default=4, help="questions answered at once (Claude)")
    parser.add_argument("--judge-concurrency", type=int, default=12, help="judge calls in flight at once")
    parser.add_argument("--judge-model", default=None, help="defaults to MEDRAG_JUDGE_MODEL")
    parser.add_argument("--judge-repeats", type=int, default=1, help="score each answer N times (judge variance)")
    parser.add_argument(
        "--record-judge",
        action="store_true",
        help="opt-in: write every judge output (claims, verdicts, reasons) to *_judge_trace.jsonl (gitignored)",
    )
    parser.add_argument(
        "--prompt", choices=["v1", "v2", "v3", "v4", "v4.1"], default="v1", help="answer prompt version"
    )
    parser.add_argument("--plan", choices=["none", "v12_t3"], default="none", help="retrieval plan preset")
    parser.add_argument("--aliases", type=Path, default=None, help="alias table for --plan (default: the only one)")
    parser.add_argument("--answerable-only", action="store_true", help="skip unanswerable questions (pilot runs)")
    parser.add_argument("--out-name", default=None, help="report file stem under data/processed/eval/reports")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing report")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
