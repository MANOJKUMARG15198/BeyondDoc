"""Retrieval planner (Retrieval v1.2, Steps 3-4): medicine resolution + intents + named sources -> a RetrievalPlan.

Rules (pure function, no I/O):
- no medicine (confidence "none")        -> standard plan: the existing hybrid search, unchanged
- ambiguous or derived                   -> soft BM25 boost for resolved medicines only; no filter, no guard
- verified                               -> boost + a medicine-filtered dense branch alongside the broad one
- verified, one medicine, no comparison, not an interaction question -> also the post-fusion identity guard
- coverage (Step 4, only when enabled and the medicine is verified):
    2+ named sources (secondary research cues count as PMC only next to another named source)
        -> one unit per (medicine, source) that has documents; sources without documents are logged
    otherwise 2+ verified medicines -> one unit per medicine (any source)
  at most MAX_UNITS units; questions without units keep the plan above unchanged.
  a question about a single study/trial with fewer than two named sources forms no per-medicine units (Step 5).
- sections (Step 5, only when enabled; applies with or without a medicine):
    soft BM25 boost on section_path phrases / section_role values mapped from the detected intents (fixed table);
    noise demotion (bibliography, back matter, letter signatures, DailyMed patient sections) unless the question asks
    for references.
Boost values, the guard cap and the coverage floor are experimental parameters, not final settings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from medical_rag.query.medicine import MedicineResolution

GUARD_EXEMPT_INTENTS = frozenset({"interactions"})
MAX_UNITS = 4
# Intent -> section_path phrases (match_phrase on the analyzed field) and section_role values; fixed, reviewable.
SECTION_PHRASES: dict[str, tuple[str, ...]] = {
    "dosage": ("dosage and administration",),
    "adverse_effects": ("adverse reactions",),
    "pharmacokinetics": ("pharmacokinetics",),
    "interactions": ("drug interactions",),
    "pregnancy_lactation": ("pregnancy", "lactation"),
    "contraindications": ("contraindications",),
    "mechanism": ("mechanism of action",),
    "ingredients_storage": ("description", "how supplied"),
    "indications": ("indications and usage",),
    "limitations": ("discussion", "limitations"),
    "overdose": ("overdosage",),
}
SECTION_ROLES: dict[str, tuple[str, ...]] = {"case_report": ("case_report",)}
NOISE_EXEMPT_INTENTS = frozenset({"bibliography"})


@dataclass(frozen=True)
class GuardSpec:
    medicine_documents: frozenset[str]
    max_other: int = 2


@dataclass(frozen=True)
class CoverageUnit:
    medicine_id: str
    source: str | None  # None: any source of this medicine
    documents: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.medicine_id}/{self.source or 'any'}"


@dataclass(frozen=True)
class RetrievalPlan:
    name: Literal["standard", "medicine", "section"] = "standard"
    boost_documents: tuple[str, ...] = ()
    boost_aliases: tuple[str, ...] = ()
    document_boost: float = 2.0
    alias_boost: float = 1.0
    medicine_dense_documents: tuple[str, ...] = ()
    guard: GuardSpec | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)
    coverage: tuple[CoverageUnit, ...] = ()
    coverage_floor: int | None = None  # None: adaptive, k // (units + 1)
    branch_size: int = 20
    branch_timeout_s: float = 0.3
    section_phrases: tuple[str, ...] = ()
    section_roles: tuple[str, ...] = ()
    section_boost: float = 1.0
    demote_noise: bool = False
    max_noise: int = 1

    @property
    def is_standard(self) -> bool:
        return not (
            self.boost_documents
            or self.boost_aliases
            or self.medicine_dense_documents
            or self.guard
            or self.coverage
            or self.section_phrases
            or self.section_roles
            or self.demote_noise
        )


STANDARD = RetrievalPlan(reasons=("no medicine resolved: standard hybrid",))


def _source(document_id: str) -> str:
    return document_id.split(":", 1)[0]


def coverage_units(
    resolution: MedicineResolution, sources: Sequence[str], single_study: bool = False
) -> tuple[list[CoverageUnit], list[str]]:
    units: list[CoverageUnit] = []
    notes: list[str] = []
    if resolution.confidence != "verified" or not resolution.medicines:
        return units, notes
    named = list(dict.fromkeys(sources))
    if len(named) >= 2:
        for med in resolution.medicines:
            for src in named:
                docs = tuple(d for d in med.documents if _source(d) == src)
                if docs:
                    units.append(CoverageUnit(med.medicine_id, src, docs))
                else:
                    notes.append(f"source_unavailable:{med.medicine_id}/{src}")
    elif len(resolution.medicines) >= 2 and single_study:
        notes.append("per-medicine units skipped: single study")
    elif len(resolution.medicines) >= 2:
        units = [CoverageUnit(m.medicine_id, None, m.documents) for m in resolution.medicines if m.documents]
    if len(units) > MAX_UNITS:
        notes.append(f"units truncated to {MAX_UNITS}")
        units = units[:MAX_UNITS]
    return units, notes


def make_plan(
    resolution: MedicineResolution,
    intents: Sequence[str] = (),
    *,
    sources: Sequence[str] = (),
    document_boost: float = 2.0,
    alias_boost: float = 1.0,
    filtered_dense: bool = True,
    guard: bool = True,
    max_other: int = 2,
    coverage: bool = False,
    coverage_floor: int | None = None,
    single_study: bool = False,
    sections: bool = False,
    section_boost: float = 1.0,
    noise: bool = False,
) -> RetrievalPlan:
    phrases = tuple(dict.fromkeys(p for i in intents for p in SECTION_PHRASES.get(i, ()))) if sections else ()
    roles = tuple(dict.fromkeys(r for i in intents for r in SECTION_ROLES.get(i, ()))) if sections else ()
    demote = noise and not (set(intents) & NOISE_EXEMPT_INTENTS)
    extra: dict = {"section_phrases": phrases, "section_roles": roles, "section_boost": section_boost,
                   "demote_noise": demote}  # fmt: skip
    section_reasons = ([f"sections={[*phrases, *roles]}"] if phrases or roles else []) + (
        ["noise demotion"] if demote else [])  # fmt: skip
    if not resolution.medicines:
        if not (phrases or roles or demote):
            return STANDARD
        return RetrievalPlan("section", reasons=("no medicine resolved", *section_reasons), **extra)
    docs = tuple(resolution.document_ids())
    aliases = tuple(sorted({m.matched_text for m in resolution.medicines}))
    reasons = [f"confidence={resolution.confidence}", f"medicines={[m.medicine_id for m in resolution.medicines]}"]
    if resolution.confidence != "verified":
        reasons.append("boost only (not verified)")
        return RetrievalPlan(
            "medicine", docs, aliases, document_boost, alias_boost, reasons=(*reasons, *section_reasons), **extra
        )
    spec = None
    exempt = sorted(set(intents) & GUARD_EXEMPT_INTENTS)
    if guard and len(resolution.medicines) == 1 and not resolution.comparison and not exempt:
        spec = GuardSpec(frozenset(docs), max_other)
        reasons.append(f"guard: at most {max_other} other-medicine passages")
    elif guard:
        reasons.append("guard skipped: " + ("comparison/multiple medicines" if not exempt else f"intent {exempt}"))
    units: list[CoverageUnit] = []
    if coverage:
        units, notes = coverage_units(resolution, sources, single_study)
        reasons += notes
        if units:
            reasons.append(f"coverage units={[u.key for u in units]} floor={coverage_floor or 'adaptive'}")
    return RetrievalPlan(
        "medicine",
        docs,
        aliases,
        document_boost,
        alias_boost,
        docs if filtered_dense else (),
        spec,
        (*reasons, *section_reasons),
        tuple(units),
        coverage_floor,
        **extra,
    )


# Experimental preset selected on the v2.0 dev split (Retrieval v1.2 arm T3) and checked once on the holdout split:
# medicine boost + filtered dense list, coverage floor 2, single-study rule, noise demotion, section boost 1.0.
# Not a final architectural decision; see docs/DECISION_LOG.md.
V12_T3: dict = {"filtered_dense": True, "guard": False, "coverage": True, "coverage_floor": 2, "noise": True,
                "sections": True, "section_boost": 1.0}  # fmt: skip


def plan_question(question: str, table, options: dict | None = None) -> RetrievalPlan:
    """Resolve medicines, intents and named sources for `question`, then build the plan (default preset: V12_T3)."""
    from medical_rag.query.intent import coverage_sources, detect_intents, refers_to_single_study
    from medical_rag.query.medicine import resolve_medicines

    return make_plan(
        resolve_medicines(question, table),
        detect_intents(question),
        sources=coverage_sources(question),
        single_study=refers_to_single_study(question),
        **(V12_T3 if options is None else options),
    )
