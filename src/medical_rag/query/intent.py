"""Fixed-lexicon intent and named-source detection (Retrieval v1.2). Several intents may apply; none means no
section preference. Sources are recorded only when the question names them, never inferred."""

from __future__ import annotations

import re

INTENTS: dict[str, str] = {
    "pregnancy_lactation": r"\b(pregnan\w*|breast-?feed\w*|lactation|nursing)\b",
    "interactions": r"\b(interact\w*|together with|taken with|taking .{1,40} with|combined with|co-?administ\w*)\b",
    "contraindications": r"\b(contraindicat\w*|should not (?:be )?(?:take|taken|use|used))\b",
    "dosage": r"\b(dose[sd]?|dosage|dosing|how much|maximum (?:daily )?dose|mg/day|titrat\w*)\b",
    "adverse_effects": r"\b(side effects?|adverse (?:effects?|reactions?|events?))\b",
    "mechanism": r"\b(how does .{1,40} work|mechanism|works? by)\b",
    "pharmacokinetics": r"\b(half-?life|clearance|absorption|absorbed|bioavailab\w*|steady state|metaboli[sz]\w*)\b",
    "liver_injury": r"\b(liver|hepat\w*|alt elevation\w*|jaundice)\b",
    "case_report": r"\b(case report\w*|case \d+|in (?:the|a) case)\b",
    "regulatory": r"\b(fda|approval|approved|supplement|nda|letter)\b",
    "ingredients_storage": r"\b(ingredients?|contains?|stored?|storage|look like|imprint\w*|debossed|markings?)\b",
    "bibliography": r"\b(references?|citations?|bibliograph\w*|which papers|published (?:studies|reports))\b",
    # Retrieval v1.2 Step 5 additions
    "indications": r"\b(approved (?:for|to treat)|indicat(?:ed|ions?)|used (?:for|to treat))\b",
    "limitations": r"\blimitations?\b",
    "overdose": r"\b(overdos\w*|much higher (?:than (?:usual|recommended) )?doses?)\b",
}
SOURCES: dict[str, str] = {
    "dailymed": r"\b(label(?:ing)?|prescribing information|package insert|medication guide|drug facts|dailymed)\b",
    "fda_drugsatfda": r"\b(fda (?:approval )?letter|approval letter|s?nda \d|supplement \d+)\b",
    "nih_livertox": r"\blivertox\b",
    "pmc_oa": r"\b(pmc|meta-?analysis|systematic review|randomi[sz]ed (?:controlled )?trial|crossover study)\b",
}
_COMPILED = {k: re.compile(v, re.I) for k, v in INTENTS.items()}
_SOURCES = {k: re.compile(v, re.I) for k, v in SOURCES.items()}


def detect_intents(question: str) -> list[str]:
    return [name for name, pattern in _COMPILED.items() if pattern.search(question)]


def detect_sources(question: str) -> list[str]:
    return [name for name, pattern in _SOURCES.items() if pattern.search(question)]


# Research wording counts as PMC only when the question also names another source (Retrieval v1.2, Step 4); on its
# own it is ignored, so single-source questions ("the label says...") are unaffected.
SECONDARY_PMC = re.compile(r"\b(stud(?:y|ies)|trials?|reviews?|articles?|papers?|meta-?analys[ie]s)\b", re.I)


def coverage_sources(question: str) -> list[str]:
    named = detect_sources(question)
    if named and "pmc_oa" not in named and SECONDARY_PMC.search(question):
        named = [*named, "pmc_oa"]
    return named


# A single study or trial ("the study", "trial sequential analysis"): with fewer than two named sources, two medicines
# in such a question are usually compared inside one paper, so per-medicine coverage units are not formed (Step 5).
SINGLE_STUDY = re.compile(r"\b(study|trial|analysis)\b", re.I)


def refers_to_single_study(question: str) -> bool:
    return bool(SINGLE_STUDY.search(question))
