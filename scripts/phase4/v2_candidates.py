"""Seeded candidate passage pools for evaluation dataset v2.0 (one file per category, never overwritten).

Rules are fixed before drafting: source and section filters per category, at least 20 tokens, and no chunk that is
gold in v1. The pool is sorted by chunk id and shuffled with random.Random(f"{seed}:{category}"); drafters take
passages in that order and log every skip with its reason (structure/formula only, packaging codes only, figure
caption only, per-document cap of 2 per category).

.venv/Scripts/python -X utf8 scripts/phase4/v2_candidates.py --category ingredients
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

CHUNKS = Path("data/processed/chunking/experiments/batch01/E3_fill/chunks.jsonl")  # the indexed chunk set
V1 = Path("data/eval/retrieval_v1.jsonl")
OUT = Path("data/eval/v2/candidates")
SEED = 20261105


def _sections(chunk: dict[str, Any]) -> str:
    return " > ".join(chunk.get("section_path") or [])


RULES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "ingredients": lambda c: c["source"] == "dailymed" and c["section_role"] == "body" and bool(
        re.search(r"inactive ingredient|active ingredient|description|how supplied|storage|package label|"
                  r"principal display", _sections(c), re.I)),
    "interactions": lambda c: c["source"] == "dailymed" and c["section_role"] in ("body", "boxed_warning") and bool(
        re.search(r"interaction|warning|precaution|contraindication|boxed|do not use|ask a doctor|stop use",
                  _sections(c), re.I)),
    "pharmacology": lambda c: c["source"] == "dailymed" and c["section_role"] == "body" and bool(
        re.search(r"pharmacolog|mechanism|pharmacokinetic|pharmacodynamic", _sections(c), re.I)),
    "pmc": lambda c: c["source"] == "pmc_oa" and c["section_role"] in ("body", "abstract") and bool(
        re.search(r"result|abstract|discussion|conclusion", _sections(c), re.I)),
    "livertox": lambda c: c["source"] == "nih_livertox" and c["section_role"] in ("body", "case_report"),
    "fda": lambda c: c["source"] == "fda_drugsatfda" and c["section_role"] == "body",
}  # fmt: skip


def build_pool(category: str, chunks: list[dict[str, Any]], excluded: set[str], seed: int = SEED) -> list[dict]:
    import random

    rule = RULES[category]
    pool = sorted(
        (c for c in chunks if c["chunk_id"] not in excluded and c["token_count"] >= 20 and rule(c)),
        key=lambda c: c["chunk_id"],
    )
    random.Random(f"{seed}:{category}").shuffle(pool)
    return [
        {
            "rank": rank,
            "chunk_id": c["chunk_id"],
            "document_id": c["document_id"],
            "title": c["title"],
            "section_path": _sections(c),
            "text_sha256": hashlib.sha256(c["text"].encode("utf-8")).hexdigest()[:16],
        }
        for rank, c in enumerate(pool)
    ]


TOPICS = {  # matched on section path plus the first 300 characters of the passage
    "renal": r"renal|kidney|creatinine clearance|dialysis",
    "hepatic": r"hepat|liver|transaminase|\bALT\b",
    "pregnancy_lactation": r"pregnan|lactation|breast|nursing",
    "interactions": r"interaction|coadminist|co-administ|concomitant",
    "dosing": r"dosage|dosing|mg/day|titrat",
    "mechanism": r"mechanism|pharmacodynamic|receptor",
    "adverse_effects": r"adverse|side effect",
    "efficacy": r"efficacy|randomi[sz]ed|placebo",
    "indication_approval": r"indicat|provides for|approved for|approval",
}


def medicine_groups(chunks: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Medicine -> documents. Brand and generic names are merged when a chunk lists exactly those two names
    (FDA letters and some labels do); a document also joins a group when the name appears in its title."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    names_by_doc: dict[str, set[str]] = {}
    titles: dict[str, str] = {}
    for c in chunks:
        names = {m.lower() for m in c.get("medicine_names") or []}
        names_by_doc.setdefault(c["document_id"], set()).update(names)
        titles[c["document_id"]] = c["title"].lower()
        if len(names) == 2:
            a, b = sorted(names)
            parent[find(b)] = find(a)
    groups: dict[str, set[str]] = {}
    members: dict[str, set[str]] = {}
    for doc, names in names_by_doc.items():
        for n in names:
            members.setdefault(find(n), set()).add(n)
            groups.setdefault(find(n), set()).add(doc)
    for root, names in members.items():
        for doc, title in titles.items():
            if any(re.search(rf"\b{re.escape(n)}\b", title) for n in names):
                groups[root].add(doc)
    label = {root: "/".join(sorted(members[root])) for root in groups}
    return {label[root]: docs for root, docs in groups.items() if len(docs) >= 2}


def topics_of(chunk: dict[str, Any]) -> set[str]:
    """Structured sources (labels, FDA letters) by section heading only; articles and LiverTox also by text."""
    text = _sections(chunk)
    if chunk["source"] in ("pmc_oa", "nih_livertox"):
        text += " " + chunk["text"][:300]
    return {t for t, pattern in TOPICS.items() if re.search(pattern, text, re.I)}


def pair_type(source_a: str, source_b: str) -> str:
    return "-".join(sorted((source_a, source_b)))


def drug_pair_units(chunks: list[dict[str, Any]], usable: list[dict], groups: dict[str, set[str]]) -> list[tuple]:
    """Two different medicines whose labels each name the other in an interactions section."""
    labels = {m: [d for d in docs if d.startswith("dailymed:")] for m, docs in groups.items()}
    inter = [c for c in usable if c["source"] == "dailymed" and "interactions" in topics_of(c)]
    units = []
    for ma, docs_a in sorted(labels.items()):
        for mb, docs_b in sorted(labels.items()):
            if ma >= mb or not docs_a or not docs_b:
                continue
            names_a, names_b = ma.split("/"), mb.split("/")
            ca = [c for c in inter if c["document_id"] in docs_a and any(n in c["text"].lower() for n in names_b)]
            cb = [c for c in inter if c["document_id"] in docs_b and any(n in c["text"].lower() for n in names_a)]
            if ca and cb:
                ca_sorted = sorted(ca, key=lambda c: c["chunk_id"])
                cb_sorted = sorted(cb, key=lambda c: c["chunk_id"])
                units.append((f"{ma} + {mb}", "interactions", ca[0]["document_id"], cb[0]["document_id"],
                              ca_sorted, cb_sorted))  # fmt: skip
    return units


def build_pair_pool(chunks: list[dict[str, Any]], excluded: set[str], seed: int = SEED) -> list[dict]:
    """Units of (medicine, document A, document B, shared topic) with one seeded passage from each document.
    Each v2 multi-document question must need both passages (evidence_rule "all")."""
    import random

    rng = random.Random(f"{seed}:multi_document")
    usable = [c for c in chunks if c["chunk_id"] not in excluded and c["token_count"] >= 40 and c["section_role"]
              in ("body", "abstract", "case_report", "boxed_warning")]  # fmt: skip
    by_doc_topic: dict[tuple[str, str], list[dict]] = {}
    for c in usable:
        for t in topics_of(c):
            by_doc_topic.setdefault((c["document_id"], t), []).append(c)
    groups = medicine_groups(chunks)
    units = drug_pair_units(chunks, usable, groups)
    for medicine, docs in sorted(groups.items()):
        ordered = sorted(docs)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                for t in sorted(TOPICS):
                    ca, cb = by_doc_topic.get((a, t)), by_doc_topic.get((b, t))
                    if ca and cb:
                        units.append((medicine, t, a, b, sorted(ca, key=lambda c: c["chunk_id"]),
                                      sorted(cb, key=lambda c: c["chunk_id"])))  # fmt: skip
    rng.shuffle(units)
    out = []
    for rank, (medicine, topic, a, b, ca, cb) in enumerate(units):
        pa, pb = rng.choice(ca), rng.choice(cb)
        out.append({
            "rank": rank, "medicine": medicine, "topic": topic,
            "pair_type": "drug-pair" if " + " in medicine else pair_type(pa["source"], pb["source"]),
            "chunk_a": pa["chunk_id"], "document_a": a, "source_a": pa["source"], "section_a": _sections(pa),
            "chunk_b": pb["chunk_id"], "document_b": b, "source_b": pb["source"], "section_b": _sections(pb),
        })  # fmt: skip
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=[*sorted(RULES), "multi_document"], required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    out = OUT / f"{args.category}.jsonl"
    if out.exists():
        print(f"refusing to overwrite {out}: pools are frozen once drafting starts", file=sys.stderr)
        return 2
    chunks = [json.loads(x) for x in CHUNKS.read_text("utf-8").splitlines() if x.strip()]
    excluded = {g for x in V1.read_text("utf-8").splitlines() if x.strip() for g in json.loads(x)["gold_chunk_ids"]}
    if args.category == "multi_document":
        pool = build_pair_pool(chunks, excluded, args.seed)
    else:
        pool = build_pool(args.category, chunks, excluded, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pool), encoding="utf-8")
    print(f"{args.category}: {len(pool)} candidates (seed {args.seed}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
