"""Text normalization shared by all source parsers.

Every string stored in a ParsedDocument passes through normalize_text, so chunk character offsets
(code points into this canonical text), token counts and coverage checks all refer to the same text.
The Unicode form is NFC, not NFKC: NFKC would rewrite symbols such as µ, ² and ½ that carry clinical
meaning.

Callers normalize small units only (one paragraph, list item or table cell), never serialized
structure, and insert separators themselves: a newline or space at every block boundary and <br/>
(S1). Superscript, subscript and citation rendering (S2, S3) also belong to the parsers.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

NORMALIZER_VERSION = "1"  # bump whenever normalize_text output can change; stored on every document

# Deleted: soft hyphen, word joiner, BOM, bidi marks/embeddings, invisible math operators (U+2061 in PMC).
_DELETE = "­⁠﻿‎‏‪‫‬‭‮⁡⁢⁣⁤"
# Turned into a space so words never merge: zero-width space and C0/C1 controls except tab, LF, CR.
# ZWJ/ZWNJ (U+200D/U+200C) are left unchanged.
_TO_SPACE = "​" + "".join(map(chr, [*range(0x00, 0x09), 0x0B, 0x0C, *range(0x0E, 0x20), *range(0x7F, 0xA0)]))
_TRANSLATION = str.maketrans(_TO_SPACE, " " * len(_TO_SPACE), _DELETE)
_TRACKED = re.compile(f"[{re.escape(_DELETE + _TO_SPACE)}�]")


def normalize_text(text: str, *, keep_newlines: bool = False, changes: Counter[str] | None = None) -> str:
    """Return NFC text with invisible characters removed and whitespace collapsed.

    With keep_newlines=True, line breaks survive (one per line, empty lines dropped); used for table
    cells and list items, whose <br/> and paragraph breaks become newlines (S1).
    If `changes` is given, every deleted or space-replaced character is counted there by code point,
    e.g. {"U+2061": 24}, and so is U+FFFD, which is kept but signals text lost before parsing.
    Whitespace collapsing (including no-break and thin spaces) is not counted.
    """
    text = unicodedata.normalize("NFC", text)
    if changes is not None:
        changes.update(f"U+{ord(ch):04X}" for ch in _TRACKED.findall(text))
    text = text.translate(_TRANSLATION)
    if not keep_newlines:
        return " ".join(text.split())
    lines = (" ".join(line.split()) for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    return "\n".join(line for line in lines if line)


def text_sha256(text: str) -> str:
    """SHA-256 hex digest of the UTF-8 text; used for Artifact.text_sha256 and content fingerprints."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
