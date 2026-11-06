"""Embed one chunking experiment with the pinned embedder (ADR-011). Resumable: re-run the same command to continue.

.venv/Scripts/python -X utf8 scripts/phase3/embed_chunks.py --experiment E3_fill
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from medical_rag.embedding.base import chunk_input  # noqa: E402
from medical_rag.embedding.config import BGE_QUERY_INSTRUCTION, EmbeddingConfig  # noqa: E402
from medical_rag.embedding.st_embedder import SentenceTransformerEmbedder  # noqa: E402
from medical_rag.embedding.store import embed_to_dir  # noqa: E402

CHUNKS = Path("data/processed/chunking/experiments/batch01")
OUT = Path("data/processed/embeddings/batch01")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--limit", type=int, default=None, help="embed only the first N chunks (smoke test)")
    args = parser.parse_args()

    chunk_file = CHUNKS / args.experiment / "chunks.jsonl"
    chunks = [json.loads(line) for line in chunk_file.read_text(encoding="utf-8").splitlines()][: args.limit]
    config_ids = sorted({c["config_id"] for c in chunks})
    config = EmbeddingConfig(query_instruction=BGE_QUERY_INSTRUCTION)
    name = args.experiment if args.limit is None else f"{args.experiment}_first{args.limit}"
    out_dir = OUT / name / config.model_name.split("/")[-1]
    print(f"{len(chunks)} chunks from {chunk_file} -> {out_dir} (embedder_id {config.embedder_id})", flush=True)

    embedder = SentenceTransformerEmbedder(config)

    def progress(done: int, total: int, seconds: float) -> None:
        print(f"  {done}/{total} chunks, {seconds / 60:.1f} min this run", flush=True)

    manifest = embed_to_dir(
        embedder,
        [c["chunk_id"] for c in chunks],
        [chunk_input(c) for c in chunks],
        out_dir,
        manifest_extra={"experiment": args.experiment, "chunk_file": str(chunk_file), "chunk_config_ids": config_ids},
        progress=progress,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "truncated_chunk_ids"}, indent=2))
    print("truncated chunks:", len(manifest["truncated_chunk_ids"]))


if __name__ == "__main__":
    sys.exit(main())
