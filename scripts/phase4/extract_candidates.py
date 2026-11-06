"""Save the fused hybrid candidate list (retrieval only, no LLM) for a v2 split, for offline reranking experiments.

Uses the unchanged retriever, index and settings (hybrid RRF, 50 + 50 candidates) and stores the whole fused list
(up to 100 chunks) per question with component ranks, RRF score and a hash of the passage text. A determinism check
compares the stored top 8 with a previous run's rows (same order expected). Never overwrites an existing file.

.venv/Scripts/python -X utf8 scripts/phase4/extract_candidates.py \
    --testset data/eval/v2/frozen/retrieval_v2.0_split75-75.jsonl --split dev \
    --compare data/processed/eval/reports/v2_dev75_run1_rows.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from medical_rag.core.config import get_settings  # noqa: E402
from medical_rag.embedding.config import BGE_QUERY_INSTRUCTION, EmbeddingConfig  # noqa: E402
from medical_rag.embedding.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from medical_rag.evaluation.manifest import run_manifest  # noqa: E402
from medical_rag.evaluation.testset_v2 import load_v2  # noqa: E402
from medical_rag.search.client import make_client  # noqa: E402
from medical_rag.search.retrieval import Retriever  # noqa: E402

OUT = Path("data/processed/eval/candidates")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--compare", type=Path, default=None, help="rows.jsonl of an earlier run (top-8 check)")
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    out = OUT / f"{args.testset.stem}_{args.split}_candidates.jsonl"
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(EmbeddingConfig(query_instruction=BGE_QUERY_INSTRUCTION))
    client = make_client(settings)
    retriever = Retriever(client, embedder, index=settings.index_alias)
    items = [i for i in load_v2(args.testset) if args.split == "all" or i.split == args.split]
    rows = []
    for item in items:
        cand = settings.retrieval_candidates
        result = retriever.search(item.question, k=2 * cand, candidates=cand)
        rows.append({
            "id": item.id,
            "question": item.question,
            "candidates": [
                {
                    "rank": r,
                    "chunk_id": h.chunk_id,
                    "document_id": h.source.get("document_id"),
                    "rrf_score": round(h.score, 6),
                    "component_ranks": h.ranks,
                    "text_sha256": hashlib.sha256(h.source.get("text", "").encode("utf-8")).hexdigest()[:16],
                }
                for r, h in enumerate(result.hits, start=1)
            ],
            "timings_ms": result.timings_ms,
        })  # fmt: skip
    mismatches = []
    if args.compare:
        lines = [json.loads(x) for x in args.compare.read_text("utf-8").splitlines() if x.strip()]
        earlier = {r["id"]: r for r in lines}
        for row in rows:
            if row["id"] in earlier:
                now = [c["chunk_id"] for c in row["candidates"][: args.k]]
                before = [h["chunk_id"] for h in earlier[row["id"]]["retrieved"]]
                if now != before:
                    mismatches.append(row["id"])
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest = run_manifest(
        testset=args.testset,
        client=client,
        alias=settings.index_alias,
        config={
            "split": args.split,
            "mode": "hybrid",
            "candidates_per_retriever": settings.retrieval_candidates,
            "embedder": embedder.config.embedder_id,
            "compare": str(args.compare) if args.compare else None,
            "top_k_mismatches": mismatches,
            "questions": len(rows),
        },  # fmt: skip
    )
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    sizes = [len(r["candidates"]) for r in rows]
    print(f"{len(rows)} questions; pool sizes {min(sizes)}-{max(sizes)}; top-{args.k} mismatches vs earlier run: "
          f"{len(mismatches)} {mismatches[:10]}")  # fmt: skip
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
