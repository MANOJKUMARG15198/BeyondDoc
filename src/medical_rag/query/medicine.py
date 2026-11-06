"""Deterministic medicine resolver (Retrieval v1.2, Step 2).

The alias table is built only from the corpus itself: each document's `medicine_names`, FDA letters (which list brand
and generic together) and LiverTox "REPRESENTATIVE TRADE NAMES" lines. A brand is linked to a generic only when one
corpus record pairs them; there is no external dictionary and no language model. Normalization (case, ®/™, a fixed
salt-suffix list, splitting combination ingredients) produces `derived` aliases; corrections come only from a
reviewed overlay whose rows are marked approved.

Resolution is a greedy longest match over aliases of at least MIN_ALIAS characters. Ambiguity, guarded common-word
brands without their stored capitalisation, and derived matches lower the confidence, which the retrieval planner
uses to decide between boosting, filtering and the unchanged standard search.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SALTS = ("hydrochloride", "hcl", "sodium", "potassium", "calcium", "besylate", "maleate", "citrate", "bitartrate",
         "phosphate", "sulfate", "polacrilex")  # fmt: skip
MIN_ALIAS = 4
_TRADE_NAMES = re.compile(r"REPRESENTATIVE TRADE NAMES\s*(.+?)\s*DRUG CLASS", re.S)
_COMPARISON = re.compile(r"\b(versus|vs\.?|compare[ds]?|comparison|difference between|differ from)\b", re.I)

Status = Literal["verified", "derived", "needs_review"]
Confidence = Literal["verified", "derived", "ambiguous", "none"]


def normalize(text: str) -> str:
    text = re.sub(r"[®™*]", "", text.lower())
    text = re.sub(r"[^a-z0-9%./+ -]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_salts(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\b(" + "|".join(SALTS) + r")\b", " ", name)).strip()


def _components(ingredient: str) -> list[str]:
    parts = [p.strip() for p in re.split(r",|\band\b", ingredient)]
    return [p for p in parts if p]


@dataclass(frozen=True)
class AliasRecord:
    alias: str
    medicine_id: str
    kind: Literal["generic", "brand", "salt_form", "combination_component", "overlay"]
    status: Status
    provenance: tuple[str, ...]  # "document_id:field"
    guard_form: str | None = None  # stored capitalisation required in the question (common-word brands)


@dataclass
class AliasTable:
    aliases: dict[str, list[AliasRecord]]
    documents: dict[str, list[str]]  # medicine_id -> document_ids
    version: str = ""

    def to_json(self) -> str:
        body = {
            "aliases": {a: [asdict(r) for r in rs] for a, rs in sorted(self.aliases.items())},
            "documents": {m: sorted(d) for m, d in sorted(self.documents.items())},
        }
        return json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> AliasTable:
        body = json.loads(text)
        aliases = {
            a: [AliasRecord(**{**r, "provenance": tuple(r["provenance"])}) for r in rs]
            for a, rs in body["aliases"].items()
        }
        return cls(aliases, body["documents"], hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])

    @classmethod
    def load(cls, path: Path) -> AliasTable:
        return cls.from_json(path.read_text(encoding="utf-8"))


def _medicine_id(ingredient: str) -> str:
    base = strip_salts(normalize(ingredient))
    parts = [strip_salts(p) for p in _components(base)]
    parts = [p for p in parts if p]
    return "+".join(sorted(parts)) if len(parts) > 1 else base


def build_alias_table(chunks: Iterable[Mapping[str, Any]], overrides: Sequence[Mapping[str, str]] = ()) -> AliasTable:
    """Aliases from the corpus; only overlay rows with status 'approved' are applied."""
    first: dict[str, Mapping[str, Any]] = {}
    product_info: dict[str, str] = {}
    for c in chunks:
        first.setdefault(c["document_id"], c)
        if c.get("source") == "nih_livertox" and (c.get("section_path") or [""])[0] == "PRODUCT INFORMATION":
            product_info[c["document_id"]] = c.get("text", "")

    aliases: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    documents: dict[str, set[str]] = defaultdict(set)

    def add(alias: str, medicine_id: str, kind: str, status: str, prov: str, guard: str | None = None) -> None:
        if len(alias) < MIN_ALIAS or alias.replace(" ", "").isdigit():
            return
        rec = aliases[alias].setdefault((medicine_id, kind), {"status": status, "prov": set(), "guard": guard})
        if status == "verified":
            rec["status"] = "verified"
        rec["prov"].add(prov)

    for doc, c in sorted(first.items()):
        names = [n for n in (c.get("medicine_names") or []) if n and n.strip()]
        if not names:
            continue
        ingredient = names[-1]  # labels and FDA letters store (brand, ingredient); single names are the ingredient
        mid = _medicine_id(ingredient)
        documents[mid].add(doc)
        ing_norm = normalize(ingredient)
        add(ing_norm, mid, "generic", "verified", f"{doc}:medicine_names")
        if strip_salts(ing_norm) != ing_norm:
            add(strip_salts(ing_norm), mid, "salt_form", "derived", f"{doc}:salt_stripped")
        comps = [strip_salts(p) for p in _components(strip_salts(ing_norm))]
        if len(comps) > 1:  # combination products: components only from the ingredient name, never the brand
            for comp in comps:
                add(comp, mid, "combination_component", "derived", f"{doc}:combination")
        for brand in names[:-1]:
            b = normalize(brand)
            if b != ing_norm:
                add(b, mid, "brand", "verified", f"{doc}:medicine_names")
        trade = _TRADE_NAMES.search(product_info.get(doc, ""))
        for brand in re.findall(r"([A-Z][A-Za-z]+)®", trade.group(1) if trade else ""):
            add(normalize(brand), mid, "brand", "verified", f"{doc}:livertox_trade_names")

    for row in overrides:
        if (row.get("status") or "").strip().lower() != "approved":
            continue
        action, alias = row["action"].strip(), normalize(row["alias"])
        if action == "add_alias":
            mid = row["medicine_id"].strip()
            add(alias, mid, "overlay", "verified", f"overlay:{row.get('note', '')}")
            if row.get("document_id"):
                documents[mid].add(row["document_id"].strip())
        elif action == "guard":
            for rec in aliases.get(alias, {}).values():
                rec["guard"] = row["guard_form"].strip()
        elif action == "exclude_alias":
            aliases.pop(alias, None)
        else:
            raise ValueError(f"unknown overlay action {action!r}")

    table = {
        a: [
            AliasRecord(a, mid, kind, rec["status"], tuple(sorted(rec["prov"])), rec["guard"])
            for (mid, kind), rec in sorted(recs.items())
        ]
        for a, recs in aliases.items()
    }
    built = AliasTable(table, {m: sorted(d) for m, d in documents.items()})
    return AliasTable.from_json(built.to_json())  # version = hash of the canonical JSON


def load_overrides(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class ResolvedMedicine:
    medicine_id: str
    matched_text: str
    span: tuple[int, int]
    confidence: Literal["verified", "derived"]
    documents: tuple[str, ...]


@dataclass
class MedicineResolution:
    medicines: list[ResolvedMedicine] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    comparison: bool = False
    confidence: Confidence = "none"
    version: str = ""

    def document_ids(self) -> list[str]:
        return sorted({d for m in self.medicines for d in m.documents})


def resolve_medicines(question: str, table: AliasTable) -> MedicineResolution:
    text = normalize(question)
    taken: list[tuple[int, int]] = []
    found: list[ResolvedMedicine] = []
    ambiguous: list[str] = []
    for alias in sorted(table.aliases, key=lambda a: (-len(a), a)):
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", text):
            span = (m.start(), m.end())
            if any(s < span[1] and span[0] < e for s, e in taken):
                continue
            taken.append(span)
            records = table.aliases[alias]
            if any(r.guard_form for r in records) and not any(
                r.guard_form and re.search(r"\b" + re.escape(r.guard_form) + r"\b", question) for r in records
            ):
                ambiguous.append(alias)
                continue
            mids = sorted({r.medicine_id for r in records})
            if len(mids) > 1:
                ambiguous.append(alias)
                continue
            conf = "verified" if any(r.status == "verified" for r in records) else "derived"
            found.append(ResolvedMedicine(mids[0], alias, span, conf, tuple(table.documents.get(mids[0], ()))))
    found.sort(key=lambda r: r.span)
    unique: dict[str, ResolvedMedicine] = {}
    for r in found:  # one entry per medicine; keep the strongest confidence
        if r.medicine_id not in unique or (
            r.confidence == "verified" and unique[r.medicine_id].confidence != "verified"
        ):
            unique[r.medicine_id] = r
    meds = list(unique.values())
    if ambiguous:
        confidence: Confidence = "ambiguous"
    elif not meds:
        confidence = "none"
    elif all(m.confidence == "verified" for m in meds):
        confidence = "verified"
    else:
        confidence = "derived"
    comparison = len(meds) >= 2 or bool(_COMPARISON.search(question))
    return MedicineResolution(meds, sorted(set(ambiguous)), comparison, confidence, table.version)
