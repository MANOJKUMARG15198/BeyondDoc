"""Build a corrected, separately versioned test set from a base set and a reviewed change log.

The base file is read only; its sha256 is checked before and after. Only rows marked "approved" in the change log
are applied. --dry-run validates every row (including proposed ones) without writing anything.

.venv/Scripts/python -X utf8 scripts/phase3/revise_testset.py --base data/eval/retrieval_v1.jsonl \
    --changes data/eval/retrieval_v1.1_changes.csv --out data/eval/retrieval_v1.1.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

from medical_rag.evaluation.manifest import sha256_file
from medical_rag.evaluation.testset import load_testset, write_testset
from medical_rag.evaluation.testset_revision import RevisionError, apply_changes, parse_changes

CHUNKS = Path("data/processed/chunking/experiments/batch01/E3_fill/chunks.jsonl")  # the indexed chunk set


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--changes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--dry-run", action="store_true", help="validate all rows as if approved; write nothing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.resolve() == args.base.resolve():
        print("refusing: --out must differ from --base", file=sys.stderr)
        return 2
    if args.out.exists() and not args.overwrite and not args.dry_run:
        print(f"refusing to overwrite {args.out}; pass --overwrite", file=sys.stderr)
        return 2
    base_sha = sha256_file(args.base)
    items = load_testset(args.base)
    chunks = [json.loads(x) for x in args.chunks.read_text("utf-8").splitlines() if x.strip()]
    with args.changes.open(encoding="utf-8-sig", newline="") as f:
        changes = parse_changes(csv.DictReader(f))
    if args.dry_run:
        changes = [dataclasses.replace(c, status="approved") for c in changes]
    try:
        revised, log = apply_changes(items, changes, chunks)
    except RevisionError as exc:
        print(f"invalid change: {exc}", file=sys.stderr)
        return 1
    applied = [e for e in log if e["applied"]]
    for e in log:
        mark = "applied" if e["applied"] else f"skipped ({e['status'] or 'no status'})"
        print(f"  {e['id']:5} {e['action']:21} {mark} {e['resolved_chunk_ids'] or ''}")
    print(f"{len(applied)} of {len(log)} changes {'valid' if args.dry_run else 'applied'}; {len(revised)} items")
    if args.dry_run:
        return 0
    write_testset(args.out, revised)
    log_path = args.out.with_name(args.out.stem + "_applied.csv")
    with log_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0]) if log else ["id"])
        writer.writeheader()
        writer.writerows({**e, "resolved_chunk_ids": " ".join(e["resolved_chunk_ids"])} for e in log)
    if sha256_file(args.base) != base_sha:
        print("ERROR: base test set changed during the run", file=sys.stderr)
        return 1
    print(f"wrote {args.out} and {log_path}; base {args.base} unchanged (sha256 {base_sha[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
