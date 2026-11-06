"""Retrieval latency benchmark: plain hybrid (A0) vs the Retrieval v1.2 preset (T3). Retrieval only, no LLM.

All 150 frozen v2.0 questions (text only; no labels are used), sequential, one warm-up pass not measured, then
`--repeats` measured passes with the arm order alternated per question to spread drift. Reports P50/P95/P99 of the
total and of each component. Single local OpenSearch node and a laptop CPU: a relative comparison, not an SLO.

.venv/Scripts/python -X utf8 scripts/phase5/bench_retrieval.py --out-name bench_run1
"""

from __future__ import annotations

import argparse
import glob
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
from medical_rag.evaluation.manifest import run_manifest  # noqa: E402
from medical_rag.evaluation.testset_v2 import load_v2  # noqa: E402
from medical_rag.query.medicine import AliasTable  # noqa: E402
from medical_rag.search.client import make_client  # noqa: E402
from medical_rag.search.planner import plan_question  # noqa: E402
from medical_rag.search.retrieval import Retriever  # noqa: E402

TESTSET = Path("data/eval/v2/frozen/retrieval_v2.0_split75-75.jsonl")
OUT = Path("data/processed/eval/v12")
K = 8


def pct(values: list[float]) -> dict:
    q = statistics.quantiles(values, n=100)
    return {"p50": round(q[49], 1), "p95": round(q[94], 1), "p99": round(q[98], 1), "n": len(values)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    out = OUT / args.out_name
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    table = AliasTable.load(Path(sorted(glob.glob("data/processed/medicine/aliases_*.json"))[0]))
    settings = get_settings()
    client = make_client(settings)
    embedder = SentenceTransformerEmbedder(EmbeddingConfig(query_instruction=BGE_QUERY_INSTRUCTION))
    retriever = Retriever(client, embedder, index=settings.index_alias)
    questions = [i.question for i in load_v2(TESTSET)]

    def run(question: str, arm: str) -> dict[str, float]:
        plan = plan_question(question, table) if arm == "T3" else None
        return retriever.search(question, k=K, candidates=settings.retrieval_candidates, plan=plan).timings_ms

    for q in questions:  # warm-up pass
        run(q, "A0")
        run(q, "T3")
    timings: dict[str, dict[str, list[float]]] = {"A0": defaultdict(list), "T3": defaultdict(list)}
    for r in range(args.repeats):
        for n, q in enumerate(questions):
            order = ("A0", "T3") if (n + r) % 2 == 0 else ("T3", "A0")
            for arm in order:
                for key, value in run(q, arm).items():
                    timings[arm][key].append(value)
    report = {
        "arms": {arm: {key: pct(v) for key, v in sorted(t.items())} for arm, t in timings.items()},
        "questions": len(questions),
        "repeats": args.repeats,
        "note": "sequential, single local OpenSearch node, laptop CPU; relative comparison only",
        "manifest": run_manifest(
            testset=TESTSET,
            client=client,
            alias=settings.index_alias,
            config={"k": K, "candidates": settings.retrieval_candidates, "alias_table_version": table.version},
        ),  # fmt: skip
    }
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for arm in ("A0", "T3"):
        print(arm, {k: v for k, v in report["arms"][arm].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
