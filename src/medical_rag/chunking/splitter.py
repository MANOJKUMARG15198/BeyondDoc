"""Two-stage, word-safe recursive splitting (rule C, M1, M3).

Stage A (`pieces`) finds boundaries in order: block, line, sentence, clause (; :), phrase (,), word, and only for
a single over-long word the counter's safe cut (punctuation, WordPiece word start, subword). Stage B (`pack`)
greedily recombines consecutive pieces, in source order, into the largest spans whose complete text
(prefix + separator + span) fits the budget. Overlap (`overlap_start`) is chosen afterwards from the previous
base span. Every piece and span is a contiguous range of the run text and only whitespace between pieces is
skipped, so nothing is truncated.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from medical_rag.chunking.sentences import sentence_ends
from medical_rag.chunking.tokenizer import ChunkingError, Counts

TOKEN_CUTS = ("punctuation", "word_start", "subword")  # finest last; any of these marks an exceptional split


@dataclass(frozen=True, slots=True)
class Piece:
    start: int
    end: int
    level: str  # finest level needed: "block", "line", "sentence", "clause", "phrase", "word" or a TOKEN_CUTS kind


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    levels: tuple[str, ...]  # levels of the pieces it packs, sorted


def _after(pattern: str) -> Callable[[str, int, int], list[int]]:
    rx = re.compile(pattern)
    return lambda text, s, e: [m.end() for m in rx.finditer(text, s, e)]


LEVELS: tuple[tuple[str, Callable[[str, int, int], list[int]]], ...] = (
    ("line", _after(r"\n")),
    ("sentence", sentence_ends),
    ("clause", _after(r"[;:](?=\s)")),
    ("phrase", _after(r",(?=\s)")),
    ("word", _after(r"\s+")),
)
_WORD_START = re.compile(r"(?<!\S)\S")


def pieces(text: str, blocks: Sequence[tuple[int, int]], counter: Counts, limit: int) -> list[Piece]:
    """Stage A over consecutive blocks text[a:b]; block boundaries are the first level."""
    return [p for a, b in blocks for p in split(text, a, b, counter, limit)]


def split(
    text: str, start: int, end: int, counter: Counts, limit: int, *, level: int = 0, used: str = "block"
) -> list[Piece]:
    """Pieces of text[start:end] that each have at most `limit` tokens."""
    if limit < 1:
        raise ChunkingError(f"token limit must be at least 1, got {limit}")
    start, end = _trim(text, start, end)
    if start >= end:
        return []
    if counter.count(text[start:end]) <= limit:
        return [Piece(start, end, used)]
    for i in range(level, len(LEVELS)):
        name, boundaries = LEVELS[i]
        cuts = [c for c in boundaries(text, start, end) if start < c < end]
        if cuts:
            edges = [start, *cuts, end]
            return [
                piece
                for a, b in zip(edges, edges[1:], strict=False)
                for piece in split(text, a, b, counter, limit, level=i + 1, used=name)
            ]
    spans = counter.split_long_word(text[start:end], limit)  # one word with no whitespace left
    kind = max((k for *_, k in spans if k in TOKEN_CUTS), key=TOKEN_CUTS.index, default="word")
    return [Piece(start + a, start + b, kind) for a, b, _ in spans]


def pack(text: str, parts: Sequence[Piece], counter: Counts, prefix: str, budget: int) -> list[Span]:
    """Stage B: greedily join consecutive pieces while count(prefix + span) stays within the budget."""
    spans: list[Span] = []
    current: tuple[int, int] | None = None
    levels: set[str] = set()
    for piece in parts:
        if current and counter.count(prefix + text[current[0] : piece.end]) <= budget:
            current = (current[0], piece.end)
            levels.add(piece.level)
            continue
        if current:
            spans.append(Span(*current, tuple(sorted(levels))))
        if counter.count(prefix + text[piece.start : piece.end]) > budget:
            raise ChunkingError(f"piece {piece.start}:{piece.end} does not fit the budget with its prefix")
        current, levels = (piece.start, piece.end), {piece.level}
    if current:
        spans.append(Span(*current, tuple(sorted(levels))))
    return spans


def overlap_start(
    text: str, previous: Span, counter: Counts, max_tokens: int, method: Literal["sentence", "word"]
) -> int:
    """Earliest whole-sentence or whole-word start inside `previous` whose tail has at most `max_tokens` tokens.

    Returns previous.end when no such tail exists (no overlap).
    """
    if max_tokens <= 0:
        return previous.end
    if method == "sentence":
        ends = sentence_ends(text, previous.start, previous.end) + _after(r"\n")(text, previous.start, previous.end)
        starts = {_trim(text, e, previous.end)[0] for e in ends}
    else:
        starts = {m.start() for m in _WORD_START.finditer(text, previous.start, previous.end)}
    best = previous.end
    for start in sorted({previous.start, *starts} - {previous.end}, reverse=True):
        if counter.count(text[start : previous.end]) > max_tokens:
            break
        best = start
    return best


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
