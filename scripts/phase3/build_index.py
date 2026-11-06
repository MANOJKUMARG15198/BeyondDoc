"""Load one embedded chunking experiment into OpenSearch and point the `medrag-chunks` alias at it.

.venv/Scripts/python -X utf8 scripts/phase3/build_index.py --experiment E3_fill
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from medical_rag.core.config import get_settings
from medical_rag.embedding.store import load_vectors
from medical_rag.search.client import cluster_health, make_client
from medical_rag.search.index import IndexSettings, build_index, index_name

CHUNKS = Path("data/processed/chunking/experiments/batch01")
EMBEDDINGS = Path("data/processed/embeddings/batch01")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", default="bge-base-en-v1.5")
    args = parser.parse_args()

    chunks = [
        json.loads(line)
        for line in (CHUNKS / args.experiment / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ids, vectors, manifest = load_vectors(EMBEDDINGS / args.experiment / args.model)
    if ids != [c["chunk_id"] for c in chunks]:
        print("chunk ids in the embeddings do not match the chunk file", file=sys.stderr)
        return 1
    (chunk_config_id,) = {c["config_id"] for c in chunks}
    client = make_client(get_settings())
    print("cluster:", cluster_health(client))
    name = index_name(chunk_config_id, manifest["embedder_id"])
    start = time.perf_counter()
    report = build_index(client, name, chunks, vectors, IndexSettings(dimension=manifest["dimension"]))
    report["seconds"] = round(time.perf_counter() - start, 1)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
