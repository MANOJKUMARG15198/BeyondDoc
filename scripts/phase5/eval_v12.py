"""Retrieval v1.2 offline comparison (retrieval only: OpenSearch + query embedding; no LLM, no judge).

Arms on the v2.0 dev split (75/75 variant; holdout untouched):
  A0  existing hybrid (no plan)
  M1  medicine BM25 boost only
  M2  M1 + medicine-filtered dense list in RRF
  M3  M2 + post-fusion identity guard (max 2 other-medicine passages)
  S1/S2/S3/SA  M2 + source/medicine coverage units with floor 1/2/3/adaptive (Step 4)
  SC  S2 + single-study rule (no per-medicine units for one-study questions)          (Step 5)
  T1  SC + noise demotion; T2 SC + intent section boost 1.0; T3 both; T4 T3 with boost 2.0
--arms selects a subset (default: all). ap_dedup@8 also reports AP with identical evidence sets counted once.
Metrics as in Phase 4 (eval_rerank.py): evidence recall@8, AP@8, MRR@8, nDCG@8, first-gold rank; wrong-medicine
passages in the top 8 (medicine identity from the v1.2 alias table); retrieval latency per question (sequential);
per-question deltas against A0 with document identities. Never overwrites an output directory.

.venv/Scripts/python -X utf8 scripts/phase5/eval_v12.py --out-name v12_run1
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from medical_rag.core.config import get_settings  # noqa: E402
from medical_rag.embedding.config import BGE_QUERY_INSTRUCTION, EmbeddingConfig  # noqa: E402
from medical_rag.embedding.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from medical_rag.evaluation.manifest import run_manifest  # noqa: E402
from medical_rag.evaluation.testset_v2 import load_v2  # noqa: E402
from medical_rag.query.intent import coverage_sources, detect_intents, refers_to_single_study  # noqa: E402
from medical_rag.query.medicine import AliasTable, resolve_medicines  # noqa: E402
from medical_rag.search.client import make_client  # noqa: E402
from medical_rag.search.planner import make_plan  # noqa: E402
from medical_rag.search.retrieval import Retriever  # noqa: E402

TESTSET = Path("data/eval/v2/frozen/retrieval_v2.0_split75-75.jsonl")
OUT = Path("data/processed/eval/v12")
K = 8
ARMS = {
    "A0": None,
    "M1": {"filtered_dense": False, "guard": False},
    "M2": {"filtered_dense": True, "guard": False},
    "M3": {"filtered_dense": True, "guard": True},
    "S1": {"filtered_dense": True, "guard": False, "coverage": True, "coverage_floor": 1},
    "S2": {"filtered_dense": True, "guard": False, "coverage": True, "coverage_floor": 2},
    "S3": {"filtered_dense": True, "guard": False, "coverage": True, "coverage_floor": 3},
    "SA": {"filtered_dense": True, "guard": False, "coverage": True, "coverage_floor": None},
    "SC": {"filtered_dense": True, "guard": False, "coverage": True, "coverage_floor": 2, "study_rule": True},
    "T1": {"filtered_dense": True, "guard": False, "coverage": True, "coverage_floor": 2, "study_rule": True,
           "noise": True},
    "T2": {"filtered_dense": True, "guard": False, "coverage": True, "coverage_floor": 2, "study_rule": True,
           "sections": True},
    "T3": {"filtered_dense": True, "guard": False, "coverage": True, "coverage_floor": 2, "study_rule": True,
           "noise": True, "sections": True},
    "T4": {"filtered_dense": True, "guard": False, "coverage": True, "coverage_floor": 2, "study_rule": True,
           "noise": True, "sections": True, "section_boost": 2.0},
}  # fmt: skip


def metrics_module():
    spec = importlib.util.spec_from_file_location("er", Path("scripts/phase4/eval_rerank.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pct(values: list[float], q: int) -> float | None:
    return round(statistics.quantiles(values, n=100)[q - 1], 1) if len(values) > 1 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--aliases", type=Path, default=None, help="alias table (default: the only one present)")
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset of arms")
    parser.add_argument("--split", default="dev", choices=["dev", "holdout"], help="holdout: final check only")
    args = parser.parse_args()
    arms = {a: ARMS[a] for a in args.arms.split(",")}
    out = OUT / args.out_name
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    alias_path = args.aliases or Path(sorted(p for p in glob.glob("data/processed/medicine/aliases_*.json"))[0])
    table = AliasTable.load(alias_path)
    doc_meds: dict[str, set[str]] = defaultdict(set)
    for medicine, docs in table.documents.items():
        for d in docs:
            doc_meds[d].add(medicine)
    m = metrics_module()
    settings = get_settings()
    client = make_client(settings)
    embedder = SentenceTransformerEmbedder(EmbeddingConfig(query_instruction=BGE_QUERY_INSTRUCTION))
    retriever = Retriever(client, embedder, index=settings.index_alias)
    items = [i for i in load_v2(TESTSET) if i.split == args.split]
    retriever.search(items[0].question, k=K, candidates=settings.retrieval_candidates)  # warm-up, not measured

    rows: dict[str, list[dict]] = {arm: [] for arm in arms}
    latency: dict[str, list[float]] = {arm: [] for arm in arms}
    for item in items:
        resolution = resolve_medicines(item.question, table)
        intents = detect_intents(item.question)
        sources = coverage_sources(item.question)
        single = refers_to_single_study(item.question)
        for arm, options in arms.items():
            plan = None
            if options is not None:
                opts = {k: v for k, v in options.items() if k != "study_rule"}
                plan = make_plan(resolution, intents, sources=sources, single_study=single and
                                 bool(options.get("study_rule")), **opts)  # fmt: skip
            result = retriever.search(item.question, k=K, candidates=settings.retrieval_candidates, plan=plan)
            latency[arm].append(result.timings_ms["total_ms"])
            top = [h.chunk_id for h in result.hits]
            docs = [h.source.get("document_id") for h in result.hits]
            rec = {"id": item.id, "category": item.category, "answerability": item.answerability,
                   "confidence": resolution.confidence, "medicines": [x.medicine_id for x in resolution.medicines],
                   "intents": intents, "plan": list(plan.reasons) if plan else [], "top8": top, "top8_docs": docs,
                   "demoted": result.selection, "total_ms": result.timings_ms["total_ms"],
                   "units": [u.key for u in plan.coverage] if plan else []}  # fmt: skip
            if item.answerability == "answerable":
                sets = item.evidence_sets()
                gold_docs = {e.document_id for e in item.evidence}
                gold_meds = set().union(*(doc_meds.get(d, set()) for d in gold_docs))
                wrong = [d for d in docs if d not in gold_docs and not (doc_meds.get(d, set()) & gold_meds)]
                rec |= {
                    "evidence_recall@8": m.recall_at_k(top, sets, K),
                    "ap@8": m.ap_at_k(top, sets),
                    "ap_dedup@8": m.ap_at_k(top, [set(x) for x in dict.fromkeys(frozenset(x) for x in sets)]),
                    "mrr@8": round(1 / r, 4) if (r := m.first_rank(top, item.gold_chunk_ids())) else 0.0,
                    "ndcg@8": m.ndcg_at_k(top, sets),
                    "first_gold_rank": m.first_rank(top, item.gold_chunk_ids()),
                    "wrong_medicine_in_top8": len(wrong),
                    "wrong_medicine_docs": sorted(set(wrong)),
                    "evidence_documents_present": round(sum(d in docs for d in gold_docs) / len(gold_docs), 4),
                }
            rows[arm].append(rec)

    keys = (
        "evidence_recall@8",
        "ap@8",
        "ap_dedup@8",
        "mrr@8",
        "ndcg@8",
        "wrong_medicine_in_top8",
        "evidence_documents_present",
    )

    def summary(group: list[dict]) -> dict:
        ans = [r for r in group if r["answerability"] == "answerable"]
        return {"questions": len(ans), **{k: round(statistics.fmean(r[k] for r in ans), 4) for k in keys}}

    report = {"alias_table": {"path": str(alias_path), "version": table.version}, "arms": {}}
    for arm in arms:
        group = rows[arm]
        cats = sorted({r["category"] for r in group if r["answerability"] == "answerable"})
        report["arms"][arm] = {
            "overall": summary(group),
            "by_category": {c: summary([r for r in group if r["category"] == c]) for c in cats},
            "first_gold_rank": dict(Counter(str(r["first_gold_rank"]) for r in group if "first_gold_rank" in r)),
            "retrieval_ms": {"p50": pct(latency[arm], 50), "p95": pct(latency[arm], 95), "p99": pct(latency[arm], 99)},
            "questions_with_units": sum(bool(r["units"]) for r in group),
            "retrieval_ms_with_units": {q: pct([r["total_ms"] for r in group if r["units"]], q) for q in (50, 95)},
        }
    base = {r["id"]: r for r in rows["A0"]}
    m2 = {r["id"]: r for r in rows["M2"]}
    for arm in [a for a in arms if a in ("S1", "S2", "S3", "SA")]:  # questions without units must match M2 exactly
        report["arms"][arm]["no_unit_questions_differing_from_M2"] = [
            r["id"] for r in rows[arm] if not r["units"] and r["top8"] != m2[r["id"]]["top8"]
        ]
    out.mkdir(parents=True, exist_ok=True)
    for arm in [a for a in arms if a != "A0"]:
        deltas = []
        for r in rows[arm]:
            if r["answerability"] != "answerable":
                continue
            b = base[r["id"]]
            d = {k: round(r[k] - b[k], 4) for k in keys}
            if any(d.values()):
                deltas.append({"id": r["id"], "category": r["category"], **{f"d_{k}": v for k, v in d.items()},
                               "added_docs": sorted(set(r["top8_docs"]) - set(b["top8_docs"])),
                               "removed_docs": sorted(set(b["top8_docs"]) - set(r["top8_docs"]))})  # fmt: skip
        report["arms"][arm]["paired_vs_A0"] = {
            "changed": len(deltas),
            "ap_improved": sum(d["d_ap@8"] > 0 for d in deltas),
            "ap_regressed": sum(d["d_ap@8"] < 0 for d in deltas),
            "recall_improved": sum(d["d_evidence_recall@8"] > 0 for d in deltas),
            "recall_regressed": sum(d["d_evidence_recall@8"] < 0 for d in deltas),
            "wrong_medicine_increased": sum(d["d_wrong_medicine_in_top8"] > 0 for d in deltas),
        }
        if deltas:
            with (out / f"paired_{arm}_vs_A0.csv").open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(deltas[0]))
                writer.writeheader()
                writer.writerows(deltas)
    report["manifest"] = run_manifest(testset=TESTSET, client=client, alias=settings.index_alias,
                                      config={"arms": arms, "k": K, "candidates": settings.retrieval_candidates,
                                              "alias_table_version": table.version})  # fmt: skip
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for arm in arms:
        (out / f"rows_{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[arm]), encoding="utf-8")
        a = report["arms"][arm]
        print(arm, a["overall"], a["retrieval_ms"], a.get("paired_vs_A0", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
