"""Validate a v2.0 draft batch and write its review CSV (read-only on the corpus and on v1).

Blocking checks (exit 1): quote not verbatim in its passage, numbers in the reference missing from the evidence,
reuse of a v1 gold chunk, duplicate ids, answerable item without evidence. Warnings: quote also present in other
documents, lexical closeness to a v1 question (Jaccard >= 0.4). The embedding-based duplicate check and the
retrieval check for unanswerable items run separately, after approval.

.venv/Scripts/python -X utf8 scripts/phase4/v2_validate.py --batch data/eval/v2/drafts/batch1_ingredients.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from medical_rag.evaluation.testset import squash
from medical_rag.evaluation.testset_v2 import check_items, load_v2, write_v2

CHUNKS = Path("data/processed/chunking/experiments/batch01/E3_fill/chunks.jsonl")
V1 = Path("data/eval/retrieval_v1.jsonl")
REVIEW = Path("data/eval/v2/review")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", squash(text)))


def nearest(question: str, others: list[dict]) -> tuple[str, float]:
    best, score = "", 0.0
    for o in others:
        a, b = _words(question), _words(o["question"])
        s = len(a & b) / max(1, len(a | b))
        if s > score:
            best, score = o["id"], s
    return best, round(score, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = REVIEW / f"{args.batch.stem}_review.csv"
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    chunks = {c["chunk_id"]: c for c in map(json.loads, CHUNKS.read_text("utf-8").splitlines())}
    v1 = [json.loads(x) for x in V1.read_text("utf-8").splitlines() if x.strip()]
    excluded = {g for i in v1 for g in i["gold_chunk_ids"]}
    items = load_v2(args.batch)
    problems = check_items(items, chunks, excluded)
    rows = []
    for item in items:
        warnings = []
        for ev in item.evidence:
            needle = squash(ev.quote)
            others = {c["document_id"] for c in chunks.values() if c["document_id"] != ev.document_id
                      and needle in squash(c["text"])}  # fmt: skip
            if others:
                warnings.append(f"quote also in {len(others)} other document(s)")
        near_id, near = nearest(item.question, v1)
        if near >= 0.4:
            warnings.append(f"lexically close to v1 {near_id} ({near})")
        first = item.evidence[0] if item.evidence else None
        rows.append({
            "id": item.id, "category": item.category, "difficulty": item.difficulty,
            "retrieval_type": item.retrieval_type, "question_style": item.question_style,
            "answerability": item.answerability, "question": item.question,
            "reference_answer": item.reference_answer,
            "gold_quotes": " || ".join(e.quote for e in item.evidence),
            "document": f"{first.document_id} v{first.document_version}" if first else "",
            "section": " || ".join(e.section_path for e in item.evidence),
            "gold_chunk_ids": " ".join(sorted(item.gold_chunk_ids())),
            "source_url": first.source_url if first else "",
            "pool_rank": item.provenance.get("pool_rank", ""),
            "blocking_problems": "; ".join(problems[item.id]) or "none",
            "warnings": "; ".join(warnings) or "none",
            "nearest_v1": f"{near_id} ({near})",
            "medical_review": item.review.medical_review,
            "reviewer_decision": "", "reviewer_edit": "", "reviewer": "",
        })  # fmt: skip
    REVIEW.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_v2(args.batch.with_name(args.batch.stem + "_resolved.jsonl"), items)
    blocking = sum(bool(p) for p in problems.values())
    print(f"{len(items)} items, {blocking} with blocking problems -> {out}")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
