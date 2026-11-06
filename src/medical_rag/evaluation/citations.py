"""Per-sentence citation coverage of a generated answer (non-LLM check).

The answerer only enforces "at least one valid citation per answer"; the system prompt asks for a citation on every
factual sentence. This check measures that: each sentence is cited, a disclaimer, a statement about what the
passages do not say, a short connective ("No."), or an uncited factual sentence. Heuristic and regex-based; its
counts are a review aid, not a judgement of medical correctness.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

CITE = re.compile(r"\[S\d+\]")
_TRAILING_CITES = re.compile(r"([.!?])\s*((?:\[S\d+\])+)")  # "... daily. [S1]" -> "... daily [S1]."
_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ABBREVIATIONS = ("e.g.", "i.e.", "vs.", "approx.", "etc.", "Fig.", "et al.")
_DOT = "․"  # one-dot leader: protects abbreviation dots while splitting
DISCLAIMER = re.compile(
    r"healthcare (professional|provider)|medical advice|pharmacist|consult (a|your) (doctor|clinician)"
    r"|general (label |product )?information",
    re.I,
)
ABSENCE = re.compile(
    r"\b(passages?|documents?|sources?|label(ing)?|guide)\b[^.]*\b(do|does|did) not\b"
    r"|\bnot (stated|described|given|mentioned|specified)\b",
    re.I,
)
MIN_FACTUAL_CHARS = 25  # shorter uncited sentences ("No.", "Partly.") are connectives, not claims


@dataclass(frozen=True)
class CitationCoverage:
    cited: list[str] = field(default_factory=list)
    uncited_factual: list[str] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)
    absence: list[str] = field(default_factory=list)
    short: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float | None:
        """Cited share of sentences that make a claim (cited + uncited factual); None when there are none."""
        n = len(self.cited) + len(self.uncited_factual)
        return round(len(self.cited) / n, 4) if n else None

    def as_dict(self) -> dict:
        return {**asdict(self), "coverage": self.coverage}


def split_sentences(text: str) -> list[str]:
    text = _TRAILING_CITES.sub(r" \2\1", text.strip())
    for abbr in _ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", _DOT))
    return [s.replace(_DOT, ".").strip() for s in _SPLIT.split(text) if s.strip()]


def citation_coverage(answer: str) -> CitationCoverage:
    out = CitationCoverage()
    for sentence in split_sentences(answer):
        if CITE.search(sentence):
            out.cited.append(sentence)
        elif DISCLAIMER.search(sentence):
            out.disclaimers.append(sentence)
        elif ABSENCE.search(sentence):
            out.absence.append(sentence)
        elif len(sentence) < MIN_FACTUAL_CHARS:
            out.short.append(sentence)
        else:
            out.uncited_factual.append(sentence)
    return out
