"""Retrieval check for v2.0 unanswerable / insufficient-evidence items (read-only; no LLM calls).

For each such item, runs BM25 and dense search (top 20 each, the unchanged retriever and index) and writes every hit
with its title, section and whether it contains any of the item's text-search patterns. A human reviews the CSV and
sets absence_verified; this script never sets it.

.venv/Scripts/python -X utf8 scripts/phase4/v2_absence_check.py --batch data/eval/v2/drafts/batch8_unanswerable.jsonl
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from medical_rag.core.config import get_settings  # noqa: E402
from medical_rag.embedding.config import BGE_QUERY_INSTRUCTION, EmbeddingConfig  # noqa: E402
from medical_rag.embedding.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from medical_rag.evaluation.testset_v2 import load_v2  # noqa: E402
from medical_rag.search.client import make_client  # noqa: E402
from medical_rag.search.retrieval import Retriever  # noqa: E402

OUT = Path("data/eval/v2/review")
TOP = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = OUT / f"{args.batch.stem}_absence_check.csv"
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(EmbeddingConfig(query_instruction=BGE_QUERY_INSTRUCTION))
    retriever = Retriever(make_client(settings), embedder, index=settings.index_alias)
    items = [i for i in load_v2(args.batch) if i.answerability != "answerable"]
    rows, summary = [], []
    for item in items:
        patterns = item.provenance.get("text_search", {}).get("patterns", [])
        target = item.provenance.get("text_search", {}).get("document")
        flagged = 0
        for mode in ("bm25", "dense"):
            result = retriever.search(item.question, mode=mode, k=TOP, candidates=settings.retrieval_candidates)
            for rank, hit in enumerate(result.hits, start=1):
                text = hit.source.get("text", "")
                matches = [p for p in patterns if re.search(re.escape(p), text, re.I)]
                flagged += bool(matches)
                rows.append({
                    "id": item.id, "answerability": item.answerability, "mode": mode, "rank": rank,
                    "chunk_id": hit.chunk_id, "document_id": hit.source.get("document_id"),
                    "is_target_document": hit.source.get("document_id") == target if target else "",
                    "title": hit.source.get("title"), "section": " > ".join(hit.source.get("section_path") or []),
                    "pattern_matches": " ".join(matches), "text_start": text[:300].replace("\n", " "),
                    "reviewer_says_answers_question": "",
                })  # fmt: skip
        summary.append((item.id, flagged))
    OUT.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for item_id, flagged in summary:
        print(f"  {item_id}: {flagged} of {2 * TOP} hits contain a search pattern")
    print(f"wrote {out} ({len(rows)} rows) for human review; absence_verified is not changed by this script")
    return 0


if __name__ == "__main__":
    sys.exit(main())
