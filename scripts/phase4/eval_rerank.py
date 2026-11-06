"""Offline retrieval comparison of reranking arms on saved hybrid candidates (no LLM, no OpenSearch, no judge).

Arms: "none" (fused order, the baseline) or a cross-encoder over the top --pool fused candidates; the final 8 passages
are the reranked top 8. Retrieval, index and fusion are not touched: candidates come from extract_candidates.py and
their passage text is re-read from the chunk store and checked against the saved text hash.

Metrics (gold = each question's evidence items; overlap copies of the same item count once):
- evidence_recall@8, candidate recall in the reranker input (the ceiling for that pool size)
- mrr@8 (first evidence chunk), ndcg@8 and ap@8 over evidence items (ap@8 is a gold-based stand-in for context
  precision; the Ragas metric itself needs the judge and runs only at Step 7)
- first-gold rank distribution, results by source and by category, multi-document coverage
- wrong-medicine passages in the final 8: from a document whose medicine group shares no medicine with the gold
  documents (questions whose gold documents have no medicine group are skipped)
- rerank latency per question: windowing, inference, total (P50/P95/P99), model load time, pairs and windows
- paired per-question changes against the baseline arm

.venv/Scripts/python -X utf8 scripts/phase4/eval_rerank.py --reranker minilm --pool 20 --out-name rr_B_minilm20
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from medical_rag.evaluation.testset_v2 import load_v2

CHUNKS = Path("data/processed/chunking/experiments/batch01/E3_fill/chunks.jsonl")
CANDIDATES = Path("data/processed/eval/candidates/retrieval_v2.0_split75-75_dev_candidates.jsonl")
TESTSET = Path("data/eval/v2/frozen/retrieval_v2.0_split75-75.jsonl")
OUT = Path("data/processed/eval/rerank")
K = 8


def covered(ranked: list[str], sets: list[set[str]], k: int) -> list[int]:
    """For each of the top-k positions, 1 if that chunk covers an evidence item not covered higher up, else 0."""
    done: set[int] = set()
    gains = []
    for chunk in ranked[:k]:
        new = [j for j, s in enumerate(sets) if j not in done and chunk in s]
        gains.append(1 if new else 0)
        done.update(new[:1])
    return gains


def ap_at_k(ranked: list[str], sets: list[set[str]], k: int = K) -> float:
    gains = covered(ranked, sets, k)
    hits, total = 0, 0.0
    for i, g in enumerate(gains, start=1):
        if g:
            hits += 1
            total += hits / i
    return round(total / min(len(sets), k), 4) if sets else 0.0


def ndcg_at_k(ranked: list[str], sets: list[set[str]], k: int = K) -> float:
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(covered(ranked, sets, k), start=1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(sets), k) + 1))
    return round(dcg / ideal, 4) if ideal else 0.0


def recall_at_k(ranked: list[str], sets: list[set[str]], k: int) -> float:
    top = set(ranked[:k])
    return round(sum(bool(s & top) for s in sets) / len(sets), 4) if sets else 0.0


def first_rank(ranked: list[str], gold: set[str], k: int = K) -> int | None:
    return next((i for i, c in enumerate(ranked[:k], start=1) if c in gold), None)


def pct(values: list[float], q: int) -> float | None:
    if not values:
        return None
    return round(statistics.quantiles(values, n=100)[q - 1], 1) if len(values) > 1 else round(values[0], 1)


def doc_medicines(chunks: list[dict]) -> dict[str, set[str]]:
    spec = importlib.util.spec_from_file_location("v2c", Path(__file__).with_name("v2_candidates.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out: dict[str, set[str]] = defaultdict(set)
    for medicine, docs in module.medicine_groups(chunks).items():
        for d in docs:
            out[d].add(medicine)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker", choices=["none", "minilm"], required=True)
    parser.add_argument("--pool", type=int, default=20, help="fused candidates passed to the reranker")
    parser.add_argument("--baseline", type=Path, default=None, help="rows.jsonl of the 'none' arm, for paired deltas")
    parser.add_argument("--out-name", required=True)
    args = parser.parse_args()
    out = OUT / args.out_name
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    chunk_list = [json.loads(x) for x in CHUNKS.read_text("utf-8").splitlines() if x.strip()]
    chunks = {c["chunk_id"]: c for c in chunk_list}
    meds = doc_medicines(chunk_list)
    items = {i.id: i for i in load_v2(TESTSET)}
    rows_in = [json.loads(x) for x in CANDIDATES.read_text("utf-8").splitlines() if x.strip()]

    reranker = None
    if args.reranker == "minilm":
        from medical_rag.search.rerank import MINILM_L6, CrossEncoderReranker

        reranker = CrossEncoderReranker(MINILM_L6)

    rows, latency = [], defaultdict(list)
    for row in rows_in:
        item = items[row["id"]]
        fused = [c["chunk_id"] for c in row["candidates"]]
        for c in row["candidates"]:  # the text scored must be the text that was retrieved
            if hashlib.sha256(chunks[c["chunk_id"]]["text"].encode("utf-8")).hexdigest()[:16] != c["text_sha256"]:
                raise RuntimeError(f"text hash mismatch for {c['chunk_id']}")
        rec = {
            "id": item.id,
            "category": item.category,
            "difficulty": item.difficulty,
            "answerability": item.answerability,
        }
        if reranker is not None:
            pool = [chunks[c] for c in fused[: args.pool]]
            result = reranker.rerank(item.question, pool)
            ranked = [r.chunk_id for r in result.ranked] + fused[args.pool :]
            for key, value in result.timings_ms.items():
                latency[key].append(value)
            rec |= {
                "pairs": result.pairs,
                "windowed_chunks": sum(r.windows > 1 for r in result.ranked),
                "forced_split_chunks": sum(r.forced_split for r in result.ranked),
                **result.timings_ms,
            }
        else:
            ranked = fused
        rec["final_top8"] = ranked[:K]
        if item.answerability == "answerable":
            sets = item.evidence_sets()
            gold = item.gold_chunk_ids()
            gold_meds = set().union(*(meds.get(e.document_id, set()) for e in item.evidence))
            wrong = None
            if gold_meds:
                gold_docs = {e.document_id for e in item.evidence}
                wrong = sum(
                    1
                    for c in ranked[:K]
                    if chunks[c]["document_id"] not in gold_docs
                    and not (meds.get(chunks[c]["document_id"], set()) & gold_meds)
                )
            rec |= {
                "source": item.evidence[0].document_id.split(":", 1)[0],
                "evidence_items": len(sets),
                "evidence_recall@8": recall_at_k(ranked, sets, K),
                "candidate_recall_in_pool": recall_at_k(fused, sets, args.pool) if reranker else None,
                "mrr@8": round(1 / r, 4) if (r := first_rank(ranked, gold)) else 0.0,
                "first_gold_rank": first_rank(ranked, gold, len(ranked)),
                "ndcg@8": ndcg_at_k(ranked, sets),
                "ap@8": ap_at_k(ranked, sets),
                "wrong_medicine_in_top8": wrong,
            }
        rows.append(rec)

    answerable = [r for r in rows if r["answerability"] == "answerable"]

    def mean(key: str, group: list[dict]) -> float | None:
        values = [r[key] for r in group if r.get(key) is not None]
        return round(statistics.fmean(values), 4) if values else None

    metrics = ("evidence_recall@8", "mrr@8", "ndcg@8", "ap@8", "candidate_recall_in_pool", "wrong_medicine_in_top8")

    def summary(group: list[dict]) -> dict:
        return {"questions": len(group), **{m: mean(m, group) for m in metrics}}

    by = {}
    for key in ("category", "source", "difficulty"):
        names = sorted({r[key] for r in answerable})
        by[key] = {name: summary([r for r in answerable if r[key] == name]) for name in names}
    ranks = Counter(str(r["first_gold_rank"]) for r in answerable)
    report = {
        "arm": {
            "reranker": args.reranker,
            "pool": args.pool if reranker else None,
            "final_passages": K,
            "model": reranker.config.__dict__ if reranker else None,
            "model_load_ms": reranker.load_ms if reranker else None,
        },
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (CANDIDATES, TESTSET)},
        "overall": summary(answerable),
        "first_gold_rank": dict(sorted(ranks.items(), key=lambda kv: (kv[0] == "None", kv[0].zfill(4)))),
        "breakdown": by,
        "rerank_latency_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99)} for k, v in latency.items()},
    }
    if reranker:
        report["windowing"] = {
            "pairs_mean": mean("pairs", rows),
            "windowed_chunks_mean": mean("windowed_chunks", rows),
            "forced_split_chunks_total": sum(r.get("forced_split_chunks", 0) for r in rows),
        }
    out.mkdir(parents=True, exist_ok=True)
    if args.baseline:
        lines = [json.loads(x) for x in args.baseline.read_text("utf-8").splitlines() if x.strip()]
        base = {b["id"]: b for b in lines}
        deltas = []
        for r in answerable:
            b = base[r["id"]]
            change = {f"d_{m}": round(r[m] - b[m], 4) for m in ("ap@8", "evidence_recall@8", "ndcg@8")}
            deltas.append({"id": r["id"], "category": r["category"], **change})
        report["paired_vs_baseline"] = {
            "improved_ap": sum(d["d_ap@8"] > 0 for d in deltas),
            "regressed_ap": sum(d["d_ap@8"] < 0 for d in deltas),
            "unchanged_ap": sum(d["d_ap@8"] == 0 for d in deltas),
            "improved_recall": sum(d["d_evidence_recall@8"] > 0 for d in deltas),
            "regressed_recall": sum(d["d_evidence_recall@8"] < 0 for d in deltas),
            "regressions": [d for d in deltas if d["d_ap@8"] < 0 or d["d_evidence_recall@8"] < 0],
        }
        with (out / "paired_vs_baseline.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(deltas[0]))
            writer.writeheader()
            writer.writerows(deltas)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("arm", "overall", "rerank_latency_ms")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
