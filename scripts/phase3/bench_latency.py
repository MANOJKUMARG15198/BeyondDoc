"""Retrieval latency and throughput on this machine: sequential P50/P95 per component, then concurrent throughput.

Queries come from the test set (question text is not printed). No LLM calls; /ask latency comes from eval_ragas.py.
.venv/Scripts/python -X utf8 scripts/phase3/bench_latency.py --testset data/eval/retrieval_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from medical_rag.core.config import get_settings  # noqa: E402
from medical_rag.embedding.config import BGE_QUERY_INSTRUCTION, EmbeddingConfig  # noqa: E402
from medical_rag.embedding.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from medical_rag.evaluation.testset import load_testset  # noqa: E402
from medical_rag.search.client import make_client  # noqa: E402
from medical_rag.search.retrieval import Retriever  # noqa: E402

REPORTS = Path("data/processed/eval/reports")


def stats(values: list[float]) -> dict[str, float]:
    q = statistics.quantiles(values, n=100)
    return {
        "p50": round(q[49], 1),
        "p95": round(q[94], 1),
        "mean": round(statistics.fmean(values), 1),
        "n": len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    questions = [i.question for i in load_testset(args.testset)]
    settings = get_settings()
    retriever = Retriever(
        make_client(settings),
        SentenceTransformerEmbedder(EmbeddingConfig(query_instruction=BGE_QUERY_INSTRUCTION)),
        index=settings.index_alias,
    )
    for q in questions[:5]:  # warm-up: model, OpenSearch caches
        retriever.search(q)

    report: dict = {
        "machine": {"cpu": platform.processor(), "logical_cpus": os.cpu_count(), "gpu": None},
        "queries": len(questions),
        "sequential": {},
        "throughput": {},
    }
    for mode in ("bm25", "dense", "hybrid"):
        timings = defaultdict(list)
        for q in questions:
            for key, value in retriever.search(q, mode=mode, k=10).timings_ms.items():
                timings[key].append(value)
        report["sequential"][mode] = {k: stats(v) for k, v in timings.items()}
        print(mode, json.dumps(report["sequential"][mode]["total_ms"]), flush=True)

    for workers in args.concurrency:
        latencies: list[float] = []

        def one(q: str, out: list[float] = latencies) -> None:
            t = time.perf_counter()
            retriever.search(q, mode="hybrid", k=10)
            out.append((time.perf_counter() - t) * 1000)

        start = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, questions))
        elapsed = time.perf_counter() - start
        report["throughput"][str(workers)] = {"qps": round(len(questions) / elapsed, 2), **stats(latencies)}
        print(f"hybrid concurrency {workers}: {report['throughput'][str(workers)]}", flush=True)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "latency_retrieval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
