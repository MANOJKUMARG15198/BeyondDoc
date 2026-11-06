"""Post-fusion evidence selection (Retrieval v1.2). Identity guard: demote, never delete.

Walking the fused list in order, passages from the resolved medicine's documents always keep their place; at most
`max_other` passages from other documents keep theirs, and the rest move (in their original order) below all of the
medicine's own passages. If the medicine's documents cannot fill the final k, the demoted passages still fill it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class HasDocument(Protocol):
    chunk_id: str
    source: dict


def apply_identity_guard(
    hits: Sequence[HasDocument], medicine_documents: frozenset[str], max_other: int = 2
) -> tuple[list[HasDocument], dict[str, str]]:
    kept: list[HasDocument] = []
    demoted: list[HasDocument] = []
    others = 0
    for hit in hits:
        if hit.source.get("document_id") in medicine_documents:
            kept.append(hit)
        elif others < max_other:
            kept.append(hit)
            others += 1
        else:
            demoted.append(hit)
    return kept + demoted, {h.chunk_id: "demoted_other_medicine" for h in demoted}


def apply_coverage_floor(
    hits: Sequence[HasDocument],
    units: Sequence[tuple[str, Sequence[str], set[str]]],
    floor: int,
    k: int = 8,
) -> tuple[list[HasDocument], dict[str, str]]:
    """Coverage-aware selection of the final k (Retrieval v1.2, Step 4).

    units: (key, branch order of chunk ids, ids in the branch's BM25 list). Each unit gets up to `floor` places for
    its best branch passages that also matched lexically (BM25), taken round-robin across units and capped at half
    of k. The remaining places go to the highest-ranked passages of `hits` (the combined fusion order). The final list
    keeps the combined order, so selection is coverage-aware while ranking stays relevance-based.
    """
    present = {h.chunk_id for h in hits}
    eligible = [[c for c in order if c in bm25 and c in present] for _, order, bm25 in units]
    reserved: dict[str, str] = {}
    cap = k // 2
    for _round in range(floor):  # one passage per unit per round, so units share the reserved places fairly
        for (key, _, _), cands in zip(units, eligible, strict=True):
            if len(reserved) >= cap:
                break
            nxt = next((c for c in cands if c not in reserved), None)
            if nxt is not None:
                reserved[nxt] = f"coverage:{key}"
    chosen = set(reserved)
    for h in hits:
        if len(chosen) >= k:
            break
        chosen.add(h.chunk_id)
    final = [h for h in hits if h.chunk_id in chosen]
    reasons = {h.chunk_id: reserved.get(h.chunk_id, "relevance") for h in final}
    return final, reasons


NOISE_ROLES = frozenset({"bibliography", "back_matter", "letter_signature"})
PATIENT_SECTION_MARKERS = ("MEDGUIDE", "MEDICATION GUIDE", "PATIENT PACKAGE INSERT", "PATIENT INFORMATION")


def is_noise(source: dict) -> bool:
    """Sections that rarely hold answer evidence (Step 5 audit): LiverTox bibliography, PMC back matter, FDA letter
    signatures, and DailyMed patient sections (Medication Guide, patient package insert, Patient Information)."""
    if source.get("section_role") in NOISE_ROLES:
        return True
    path = str(source.get("section_path") or "").upper()
    return source.get("source") == "dailymed" and any(m in path for m in PATIENT_SECTION_MARKERS)


def apply_noise_demotion(hits: Sequence[HasDocument], max_noise: int = 1) -> tuple[list[HasDocument], dict[str, str]]:
    """Demote, never delete: the first `max_noise` noise passages keep their place; the rest move (in order) below
    all other passages."""
    kept: list[HasDocument] = []
    demoted: list[HasDocument] = []
    seen = 0
    for hit in hits:
        if is_noise(hit.source):
            if seen < max_noise:
                kept.append(hit)
            else:
                demoted.append(hit)
            seen += 1
        else:
            kept.append(hit)
    return kept + demoted, {h.chunk_id: "demoted_noise" for h in demoted}
