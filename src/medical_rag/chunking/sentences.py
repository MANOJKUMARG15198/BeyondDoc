"""Sentence boundaries for the recursive splitter (P2-13 baseline: regex rules, no dependency).

A boundary is placed after ., !, ?, ... or … (plus closing quotes/brackets) when whitespace follows and the next
word can start a sentence: an uppercase letter, digit, bullet, Greek letter, or a lowercase scientific term such
as mRNA, eGFR or p-glycoprotein. A full stop is not a boundary after a known abbreviation, a dotted abbreviation
(i.v., b.i.d.), a single initial, a line-leading enumeration ("5.2.") or a number labelling a table, figure or
section. A missed boundary only makes a piece coarser while a false one cuts a sentence, so the rules lean
towards not splitting. Units (mg, mL, hr) are deliberately not abbreviations: labels do not abbreviate them
with a full stop.
"""

from __future__ import annotations

import re

ABBREVIATIONS = frozenset(
    "dr mr mrs ms prof st jr sr inc ltd co corp no nos fig figs ref refs vol vols eq al "
    "approx ca cf vs viz resp incl".split()
)
NUMBER_LABELS = frozenset("table tables figure figures fig figs section sections step steps appendix part".split())

_CANDIDATE = re.compile(r"(?:\.\.\.|[.!?…])[\"'”’»)\]]*(?=\s)")
_NEXT_WORD = re.compile(r"\s+([\"'“‘«(\[]*)(\S+)")
_DOTTED = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]")
_ENUMERATION = re.compile(r"\(?(?:\d+(?:\.\d+)*|[a-z]|x{0,3}(?:ix|iv|v?i{0,3}))\)?", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:\.\d+)*")
_LOWER_TERM = re.compile(r"[a-z]{1,3}[A-Z0-9\-‐‑]")
_GREEK = re.compile(r"[Ͱ-Ͽ]")
_WRAPPERS = "\"'“‘«”’»()[]"


def sentence_ends(text: str, start: int = 0, end: int | None = None) -> list[int]:
    """Indices just after each sentence end in text[start:end], in order."""
    end = len(text) if end is None else end
    ends = []
    for m in _CANDIDATE.finditer(text, start, end):
        nxt = _NEXT_WORD.match(text, m.end(), end)
        if nxt is None or not _starts_sentence(nxt[1], nxt[2]):
            continue
        if m[0][0] == "." and not m[0].startswith("...") and not _full_stop_ends(text, start, m.start()):
            continue
        ends.append(m.end())
    return ends


def _starts_sentence(opening: str, word: str) -> bool:
    first = word[0]
    if first.isupper() or first.isdigit() or _GREEK.match(first):
        return True
    return not opening and (first == "•" or bool(_LOWER_TERM.match(word)))


def _full_stop_ends(text: str, floor: int, dot: int) -> bool:
    """True when the full stop at `dot` ends a sentence rather than an abbreviation, initial or label."""
    line_start = max(text.rfind("\n", floor, dot) + 1, floor)
    words = text[line_start:dot].split()
    if not words:
        return False
    if len(words) == 1 and _ENUMERATION.fullmatch(words[0]):
        return False  # "1." / "5.2." / "a." / "IV." opening a line
    last = words[-1].strip(_WRAPPERS)
    if last.lower() in ABBREVIATIONS or _DOTTED.fullmatch(last) or (len(last) == 1 and last.isalpha()):
        return False
    return not (_NUMBER.fullmatch(last) and len(words) > 1 and words[-2].strip(_WRAPPERS).lower() in NUMBER_LABELS)
