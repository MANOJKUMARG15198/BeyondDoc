"""Chunking benchmark: Fill-mode overlap E1-E4 (0/32/64/96 tokens) on a document set, per source.

Fill mode is the confirmed overlap allocation; the overlap amount stays open. The other open settings are held
fixed and stated in the report as experiment settings, not decisions. Raw files are checksum-verified, read in
memory only, and re-hashed at the end to prove they are unchanged. No overlap value is ranked here: that needs
retrieval results.

.venv/Scripts/python -X utf8 scripts/phase2/chunk_sample.py --set sample20
.venv/Scripts/python -X utf8 scripts/phase2/chunk_sample.py --set batch01   (only after the sample20 review)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_tokenizers import OUT, SAMPLE, pct, peak_memory_mb

from medical_rag.chunking.chunker import chunk_document
from medical_rag.chunking.config import ChunkingConfig
from medical_rag.chunking.io import write_json, write_jsonl
from medical_rag.chunking.tokenizer import ChunkingError, TokenCounter
from medical_rag.chunking.validation import validate
from medical_rag.ingestion.registry import parse_raw
from medical_rag.ingestion.sources import iter_documents

EXPERIMENTS = Path("data/processed/chunking/experiments")
FIXED = {
    "overlap_fit": "fill",  # confirmed 2026-11-04
    "overlap_method": "sentence",
    "highlights": "exclude",  # H-1 open
    "abstracts": "include",  # P2-7 open
    "table_format": "pipe",
}
RUNS = [("E1", 0), ("E2", 32), ("E3", 64), ("E4", 96)]
SETS = {
    "sample20": lambda source, sid: (
        sid in SAMPLE.get(source, ()) or source == "nih_livertox" and sid in LIVERTOX_SAMPLE
    ),
    "batch01": lambda source, sid: True,
}
LIVERTOX_SAMPLE = {"NBK548252", "NBK548894", "NBK548236"}  # Gabapentin, Cyclobenzaprine, Atorvastatin


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True, choices=sorted(SETS))
    name = parser.parse_args().set

    t0 = time.perf_counter()
    raws = list(iter_documents(SETS[name]))
    hashes = {r.raw_file: r.raw_sha256 for r in raws}
    docs = [parse_raw(r) for r in raws]
    parse_s = time.perf_counter() - t0
    counter = TokenCounter(ChunkingConfig(overlap_tokens=0, **FIXED).tokenizer)

    lines = [
        f"# Chunking benchmark — {name} ({len(docs)} documents)",
        "",
        f"Fixed experiment settings (not decisions): {FIXED}; max_tokens 512 (provisional), reserved_special_tokens 0, "
        "tokenizer google-bert/bert-base-uncased (provisional, add_special_tokens=False).",
        f"Documents by source: {dict(Counter(d.source for d in docs))}. Loaded and parsed in {parse_s:.2f} s.",
        "",
        "| run | source | docs | chunks (text/table/notes/figure) | tokens min / median / P95 / max | "
        "text coverage | row coverage | overlap dup. | chunks with overlap | failures | errors | token splits |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    timing = []
    for run, overlap in RUNS:
        cfg = ChunkingConfig(overlap_tokens=overlap, **FIXED)
        start = time.perf_counter()
        per: dict[str, dict] = defaultdict(lambda: defaultdict(list))
        for doc in docs:
            bucket = per[doc.source]
            bucket["docs"].append(doc.document_id)
            try:
                result = chunk_document(doc, cfg, counter)
            except ChunkingError as exc:
                bucket["errors"].append(f"{doc.document_id}: {exc}")
                continue
            check = validate(doc, result, cfg, counter)
            bucket["chunks"] += result.chunks
            bucket["exclusions"] += result.exclusions
            bucket["records"].append(result.document)
            bucket["failures"] += [f"{doc.document_id}: {f}" for f in check.failures]
            bucket["coverage"].append((check.text_chars, check.covered_chars, check.table_rows, check.covered_rows))
        elapsed = time.perf_counter() - start
        timing.append(f"{run}: {elapsed:.2f} s")
        out = EXPERIMENTS / name / f"{run}_fill"
        everything = {
            k: [x for b in per.values() for x in b[k]]
            for k in ("chunks", "exclusions", "records", "errors", "failures")
        }
        write_jsonl(out / "chunks.jsonl", everything["chunks"])
        write_jsonl(out / "exclusions.jsonl", everything["exclusions"])
        write_jsonl(out / "documents.jsonl", everything["records"])
        write_json(
            out / "manifest.json",
            {
                "config": cfg.model_dump(mode="json"),
                "config_id": cfg.config_id,
                "chunks": len(everything["chunks"]),
                "errors": everything["errors"],
                "validation_failures": everything["failures"],
            },
        )
        for source, b in sorted(per.items()) + [
            ("ALL", everything | {"docs": docs, "coverage": [c for x in per.values() for c in x["coverage"]]})
        ]:
            lines.append(_row(run, overlap, source, b))
        for line in everything["errors"] + everything["failures"][:20]:
            print(run, line)

    e1 = [
        json.loads(x)
        for x in (EXPERIMENTS / name / "E1_fill" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ex = [
        json.loads(x)
        for x in (EXPERIMENTS / name / "E1_fill" / "exclusions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    unchanged = all(sha256(f) == h for f, h in hashes.items())
    lines += [
        "",
        f"Timing: {', '.join(timing)}. Peak process memory: {peak_memory_mb():.0f} MB. "
        f"Raw files unchanged after all runs: {unchanged}.",
        "",
        "## E1 detail",
        "",
        f"- Chunks per document: {dict(Counter(c['document_id'] for c in e1))}",
        f"- Section roles: {dict(Counter(c['section_role'] for c in e1))}",
        f"- Case reports: {dict(Counter(c['case_id'] for c in e1 if c['case_id']))}",
        f"- Notices: {dict(Counter(n for c in e1 for n in c['notices']))}",
        f"- Split levels: {dict(Counter(lv for c in e1 for lv in c['split_levels']))}",
        f"- Exclusions by reason: {dict(Counter(e['reason'] for e in ex))}",
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    path = OUT / f"chunking_benchmark_{name}.md"
    path.write_text(report, encoding="utf-8")
    print(f"written: {path} and {EXPERIMENTS / name}/")
    return 0 if unchanged else 1


def _row(run: str, overlap: int, source: str, b: dict) -> str:
    chunks = b["chunks"]
    tokens = sorted(c.token_count for c in chunks) or [0]
    kinds = Counter(c.kind for c in chunks)
    text_chars, covered, rows, covered_rows = (
        (sum(x) for x in zip(*b["coverage"], strict=False)) if b["coverage"] else (0, 0, 0, 0)
    )
    dup = sum(c.overlap_tokens for c in chunks) / max(sum(tokens), 1)
    splits = sum("token_split" in c.notices for c in chunks)
    return (
        f"| {run} ({overlap}) | {source} | {len(b['docs'])} | {len(chunks)} "
        f"({kinds['text']}/{kinds['table']}/{kinds['table_notes']}/{kinds['figure']}) "
        f"| {tokens[0]} / {statistics.median(tokens):.0f} / {pct(tokens, 0.95)} / {tokens[-1]} "
        f"| {covered / max(text_chars, 1):.2%} | {f'{covered_rows / rows:.2%}' if rows else '—'} | {dup:.2%} "
        f"| {sum(c.overlap_tokens > 0 for c in chunks)} | {len(b['failures'])} | {len(b['errors'])} | {splits} |"
    )


if __name__ == "__main__":
    sys.exit(main())
