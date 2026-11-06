"""Build the versioned medicine alias table (Retrieval v1.2) from the indexed chunk set and the reviewed overlay.

Writes data/processed/medicine/aliases_<version>.json and a review CSV of every derived or overlay alias. Only overlay
rows marked approved are applied. Never overwrites an existing version.

.venv/Scripts/python -X utf8 scripts/phase5/build_medicine_aliases.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from medical_rag.query.medicine import build_alias_table, load_overrides

CHUNKS = Path("data/processed/chunking/experiments/batch01/E3_fill/chunks.jsonl")
OUT = Path("data/processed/medicine")
OVERRIDES = OUT / "medicine_aliases_overrides.csv"


def main() -> int:
    chunks = [json.loads(x) for x in CHUNKS.read_text("utf-8").splitlines() if x.strip()]
    overrides = load_overrides(OVERRIDES)
    table = build_alias_table(chunks, overrides)
    out = OUT / f"aliases_{table.version}.json"
    if out.exists():
        print(f"{out} already exists (same content); nothing written")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(table.to_json(), encoding="utf-8")
    review = OUT / f"aliases_{table.version}_review.csv"
    with review.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alias", "medicine_id", "kind", "status", "provenance", "documents", "reviewer_decision"])
        for alias, records in sorted(table.aliases.items()):
            for r in records:
                if r.status != "verified" or r.kind == "overlay":
                    writer.writerow([alias, r.medicine_id, r.kind, r.status, " ".join(r.provenance),
                                     len(table.documents.get(r.medicine_id, [])), ""])  # fmt: skip
    applied = sum((o.get("status") or "").lower() == "approved" for o in overrides)
    print(f"{len(table.aliases)} aliases, {len(table.documents)} medicines, overlay rows applied {applied} "
          f"of {len(overrides)} -> {out}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
