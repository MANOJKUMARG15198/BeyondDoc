"""BERT-uncased token counting for chunking (P2-1, provisional).

Counts with add_special_tokens=False and with truncation and padding disabled, and finds safe cut positions
inside a single over-long word. It never changes, stores or embeds text.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

from tokenizers import Tokenizer

from medical_rag.chunking.config import TokenizerSpec

# Cut points inside one word: after inner punctuation, or before an opening bracket.
_INNER_BREAK = re.compile(r"(?<=[-‐-—/,;:)\]}])(?=.)|(?<=.)(?=[(\[{])")


class ChunkingError(ValueError):
    """Content cannot be split into chunks that fit the ceiling without losing text."""


class Counts(Protocol):
    """What the splitter, tables and chunker need from a counter (TokenCounter, or a test double)."""

    def count(self, text: str) -> int: ...

    def split_long_word(self, text: str, limit: int) -> list[tuple[int, int, str]]: ...


class TokenCounter:
    def __init__(self, spec: TokenizerSpec) -> None:
        data = spec.file.read_bytes()
        if hashlib.sha256(data).hexdigest() != spec.sha256:
            raise ChunkingError(f"tokenizer file is not {spec.name}@{spec.revision}: {spec.file}")
        self._tok = Tokenizer.from_str(data.decode("utf-8"))
        self._tok.no_truncation()
        self._tok.no_padding()
        self.spec = spec

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def safe_cut(self, text: str, limit: int) -> tuple[int, str]:
        """Return (index, kind) so that text[:index] has at most `limit` tokens; index is 0 < index <= len(text).

        kind is "none" when the whole text fits, else the safest boundary found: "punctuation" (inside a
        compound word), "word_start" (a WordPiece word boundary) or "subword" (inside a word: exceptional).
        """
        if limit < 1:
            raise ChunkingError(f"token limit must be at least 1, got {limit}")
        enc = self._tok.encode(text, add_special_tokens=False)
        if len(enc.ids) <= limit:
            return len(text), "none"
        reach = enc.offsets[limit][0]  # a prefix ending later already holds more than `limit` original tokens
        tiers = (
            ("punctuation", [m.start() for m in _INNER_BREAK.finditer(text)]),
            ("word_start", [s for (s, _), t in zip(enc.offsets, enc.tokens, strict=True) if not t.startswith("##")]),
            ("subword", [s for s, _ in enc.offsets]),
        )
        for kind, candidates in tiers:
            for cut in sorted({c for c in candidates if 0 < c <= reach}, reverse=True):
                if self.count(text[:cut]) <= limit:  # re-tokenizing a prefix can differ: always verify
                    return cut, kind
        raise ChunkingError(f"no safe cut within {limit} tokens for text starting {text[:40]!r}")

    def split_long_word(self, text: str, limit: int) -> list[tuple[int, int, str]]:
        """Cover all of `text` with (start, end, kind) spans that each fit `limit`; nothing is dropped."""
        spans, start = [], 0
        while start < len(text):
            cut, kind = self.safe_cut(text[start:], limit)
            spans.append((start, start + cut, kind))
            start += cut
        return spans
