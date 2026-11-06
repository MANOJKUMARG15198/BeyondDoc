"""Inline text extraction for JATS-family XML (PMC JATS, LiverTox NXML), applying ADR-010 S1-S3.

S1: <break/> and nested block elements become separators. S2: a number superscript right after a digit becomes a
caret (10^9); other sup/sub join plainly with the two approved layout repairs (shared with the SPL parser, so
the rules cannot drift apart). S3: a citation the source does not bracket is written as [n], whether it is a bare
<xref ref-type="bibr"> or a superscript group of them. Table-footnote references become "(footnote a)" markers
so chunking keeps notes with their rows. E-mail addresses are never kept (J-2). Formulas keep their source text
and are counted; a dedicated rule is deferred (J-1).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

from medical_rag.ingestion.spl import MARKER_GAP, NUMBER, WORD_START, _drop_layout_gap, _last_char
from medical_rag.ingestion.text import normalize_text

MATHML = "{http://www.w3.org/1998/Math/MathML}"
XLINK = "{http://www.w3.org/1999/xlink}"
CITATION_TEXT = re.compile(r"[\d\s,;–\-]+")
NESTED = {"p", "list", "list-item", "title", "caption", "def-item", "term", "def", "disp-quote", "statement"}
FLATTENED = {"table-wrap", "table", "fig", "boxed-text", "tr"}  # block structure met inside inline content
FORMULAS = {"inline-formula", "disp-formula"}
CITATION_PARTS = {  # adjacent in structured citations with no whitespace between them
    "name", "string-name", "surname", "given-names", "person-group", "collab", "etal", "article-title",
    "chapter-title", "source", "year", "volume", "issue", "fpage", "lpage", "elocation-id", "pub-id", "edition",
    "publisher-name", "publisher-loc", "comment", "date-in-citation", "uri", "series",
}  # fmt: skip


@dataclass(slots=True)
class InlineContext:
    changes: Counter[str] = field(default_factory=Counter)
    formulas: int = 0
    emails_removed: int = 0
    references: list[str] = field(default_factory=list)  # fig/table ids cited while building the current block

    def norm(self, text: str, *, keep_newlines: bool = False) -> str:
        return MARKER_GAP.sub(r"\1", normalize_text(text, keep_newlines=keep_newlines, changes=self.changes))


def local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def inline(el: ET.Element, ctx: InlineContext, out: list[str], skip: frozenset[str] = frozenset()) -> None:
    """Append el's text in document order; children named in `skip` are left to the caller (their tail is kept)."""
    if el.text:
        out.append(el.text)
    for child in el:
        inline_child(child, ctx, out, skip)


def inline_child(child: ET.Element, ctx: InlineContext, out: list[str], skip: frozenset[str] = frozenset()) -> None:
    """Append one child element and its tail."""
    name = local(child.tag)
    if name in skip:
        pass
    elif name == "break":
        out.append("\n")
    elif name in ("sup", "sub"):
        _script(child, name, ctx, out)
    elif name == "xref":
        _xref(child, ctx, out)
    elif name == "email":
        ctx.emails_removed += 1
        out.append("[e-mail address removed]")
    elif name in FORMULAS or (isinstance(child.tag, str) and child.tag.startswith(MATHML)):
        ctx.formulas += 1
        out.append(" " + "".join(child.itertext()) + " ")
    elif name in NESTED or name in FLATTENED:
        out.append("\n")
        inline(child, ctx, out)
        out.append("\n")
    elif name in ("td", "th") or name in CITATION_PARTS:
        out.append(" ")
        inline(child, ctx, out)
        out.append(" ")
    else:  # italic, bold, sc, ext-link, uri, named-content, styled-content, underline, monospace, label...
        inline(child, ctx, out)
    if child.tail:
        out.append(child.tail)


def text_of(el: ET.Element | None, ctx: InlineContext, *, keep_newlines: bool = False) -> str:
    if el is None:
        return ""
    out: list[str] = []
    inline(el, ctx, out)
    return ctx.norm("".join(out), keep_newlines=keep_newlines)


def _script(child: ET.Element, name: str, ctx: InlineContext, out: list[str]) -> None:
    xrefs = [x for x in child if local(x.tag) == "xref"]
    inner = "".join(child.itertext())
    if xrefs and all(x.get("ref-type") == "bibr" for x in xrefs) and CITATION_TEXT.fullmatch(inner):
        out.append(f"[{' '.join(inner.split())}]")  # superscript citation group: S3
    elif xrefs:
        inline(child, ctx, out)  # e.g. a table-footnote marker inside <sup>
    else:
        if name == "sub":
            _drop_layout_gap(out)
        if name == "sup" and NUMBER.fullmatch(inner.strip()) and _last_char(out).isdigit():
            out.append("^" + inner.strip())
        else:
            out.append(inner)
    if WORD_START.match(child.tail or ""):
        out.append(" ")


def _xref(child: ET.Element, ctx: InlineContext, out: list[str]) -> None:
    kind, inner = child.get("ref-type"), "".join(child.itertext()).strip()
    if kind in ("fig", "table") and child.get("rid"):
        ctx.references.extend(child.get("rid", "").split())
    if kind == "table-fn":
        out.append(f" (footnote {inner or child.get('rid', '?')}) ")
    elif kind == "bibr" and inner and not _bracketed(out, child.tail):
        out.append(f"[{inner}]")
    else:
        out.append(inner)


def _bracketed(out: list[str], tail: str | None) -> bool:
    before = "".join(out[-3:]).rstrip()[-1:]
    after = (tail or "").lstrip()[:1]
    return before in ("[", "(", ",", ";", "–", "-") or after in ("]", ")", ",", ";", "–", "-")
