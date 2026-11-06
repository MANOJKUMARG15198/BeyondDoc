"""Retrieval evaluation on the test set: BM25 vs dense vs hybrid (RRF), query instruction with vs without.

Reports hit/recall/MRR/nDCG at 1/3/5/10, per source, and per-query latency P50/P95 of each component.
.venv/Scripts/python -X utf8 scripts/phase3/eval_retrieval.py --testset data/eval/retrieval_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from medical_rag.core.config import get_settings  # noqa: E402
from medical_rag.embedding.config import BGE_QUERY_INSTRUCTION, EmbeddingConfig  # noqa: E402
from medical_rag.embedding.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from medical_rag.evaluation.retrieval_metrics import score_run  # noqa: E402
from medical_rag.evaluation.testset import load_testset  # noqa: E402
from medical_rag.search.client import make_client  # noqa: E402
from medical_rag.search.retrieval import Retriever  # noqa: E402

REPORTS = Path("data/processed/eval/reports")


def pct(values: list[float], q: float) -> float:
    return round(statistics.quantiles(values, n=100)[int(q) - 1], 1) if len(values) > 1 else (values or [0])[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--only-approved", action="store_true", help="skip items not marked approved/edited")
    args = parser.parse_args()

    items = [i for i in load_testset(args.testset) if i.kind == "answerable" and i.review_status != "rejected"]
    if args.only_approved:
        items = [i for i in items if i.review_status in ("approved", "edited")]
    settings = get_settings()
    client = make_client(settings)
    base = SentenceTransformerEmbedder(EmbeddingConfig(query_instruction=BGE_QUERY_INSTRUCTION))
    runs = {
        "bm25": ("bm25", base),
        "dense+instr": ("dense", base),
        "hybrid+instr": ("hybrid", base),
    }
    plain = base.with_query_instruction(None)
    runs |= {"dense": ("dense", plain), "hybrid": ("hybrid", plain)}

    report: dict = {"testset": str(args.testset), "queries": len(items), "index": settings.index_alias, "runs": {}}
    for name, (mode, embedder) in runs.items():
        retriever = Retriever(client, embedder, index=settings.index_alias)
        rows, rows_equiv, by_source, timings = [], [], defaultdict(list), defaultdict(list)
        for item in items:
            result = retriever.search(item.question, mode=mode, k=10, candidates=settings.retrieval_candidates)
            ranked, gold = [h.chunk_id for h in result.hits], set(item.gold_chunk_ids)
            rows.append((ranked, gold))
            rows_equiv.append((ranked, item.gold_with_equivalents()))
            by_source[item.source].append((ranked, gold))
            for key, value in result.timings_ms.items():
                timings[key].append(value)
        report["runs"][name] = {
            "overall": score_run(rows),
            "by_source": {s: score_run(r, ks=(5, 10)) for s, r in sorted(by_source.items())},
            "latency_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95)} for k, v in timings.items()},
        }
        if any(i.alt_gold_chunk_ids for i in items):  # v1.1+: strict gold above, gold with equivalents here
            report["runs"][name]["overall_with_equivalents"] = score_run(rows_equiv)
        o = report["runs"][name]["overall"]
        print(f"{name:13} hit@5 {o['hit@5']:.3f} recall@10 {o['recall@10']:.3f} mrr@10 {o['mrr@10']:.3f} "
              f"ndcg@10 {o['ndcg@10']:.3f} | total p50 {report['runs'][name]['latency_ms']['total_ms']['p50']} ms",
              flush=True)  # fmt: skip
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"retrieval_{args.testset.stem}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
