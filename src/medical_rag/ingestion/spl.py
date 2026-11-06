"""DailyMed SPL XML parser: one SPL document -> ParsedDocument (pure: bytes in, no file access).

Content is tagged with a SectionRole and never excluded here; the chunk filters apply the content rules.
Text extraction follows ADR-010: S1 a separator at every block boundary and <br/> (newline in table cells
and list items, space elsewhere); S2 a caret for a number superscript directly after a digit (10^9), other
sup/sub joined plainly; S4 list items carry their nesting level. Two source-layout repairs keep words
apart and formulas whole: a space is added after sup/sub when a word follows directly ("B<sub>12</sub>levels"),
and indentation containing a line break before a <sub> is dropped ("C\n   <sub>18</sub>H" -> "C18H").
Footnotes become inline markers "(footnote N)"; their text is kept with the table that holds them, or as a
paragraph after the block.
Images keep only their alt text (no OCR).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

from medical_rag.ingestion.models import (
    Block,
    BlockKind,
    HeaderSource,
    ListItem,
    ParsedDocument,
    Section,
    SectionRole,
    Table,
    TitleSource,
)
from medical_rag.ingestion.table_heuristics import header_leaves_no_data, infer_header_rows, is_layout
from medical_rag.ingestion.text import NORMALIZER_VERSION, normalize_text

NS = "{urn:hl7-org:v3}"
SOURCE_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={}"
NDC_SYSTEM = "2.16.840.1.113883.6.69"
ROLES = {
    "34066-1": SectionRole.BOXED_WARNING,
    "51945-4": SectionRole.PACKAGE_LABEL,
    "48780-1": SectionRole.PRODUCT_DATA,
}
HIGHLIGHTS_TITLE = re.compile(r"(these highlights do not include|highlights of prescribing information)", re.I)
NOT_A_TITLE = re.compile(  # printed titles that name no product: NDC codes, generic headings
    r"^(\d{4,5}-\d{3,4}(-\d{1,2})?|drug facts|package insert|prescribing information|label|spl unclassified)$",
    re.IGNORECASE,
)
SECTION_NUMBER = re.compile(r"^(\d{1,2}(?:\s*\.\s*\d{1,2}){0,3})\.?\s+(\S.*)$")
NUMBER = re.compile(r"[-−]?\d+(?:\.\d+)?")
WORD_START = re.compile(r"[a-z]|[A-Z][a-z]{2}")  # a word, not a chemical element symbol such as H, O or Na
INLINE = {"content", "linkHtml", "sub", "sup", "br", "footnote", "footnoteRef", "renderMultiMedia"}
NESTED_BLOCKS = {"paragraph", "item", "caption"}  # block elements met inside inline content
MAX_SPAN = 50
MARKER_GAP = re.compile(r"(\(footnote \d+\)) (?=[.,;:!?)\]])")  # "food (footnote 1) ." -> "food (footnote 1)."


@dataclass(slots=True)
class _Context:
    media: dict[str, ET.Element]  # observationMedia ID -> element
    changes: Counter[str] = field(default_factory=Counter)
    unknown: Counter[str] = field(default_factory=Counter)
    footnote_numbers: dict[str, int] = field(default_factory=dict)
    footnotes: list[str] = field(default_factory=list)  # footnote texts met while building the current block
    figures: list[ET.Element] = field(default_factory=list)  # renderMultiMedia met while building the current block

    def norm(self, text: str, *, keep_newlines: bool = False) -> str:
        text = normalize_text(text, keep_newlines=keep_newlines, changes=self.changes)
        return MARKER_GAP.sub(r"\1", text)

    def footnote_number(self, ident: str | None) -> int:
        key = ident or f"anonymous-{len(self.footnote_numbers)}"
        return self.footnote_numbers.setdefault(key, len(self.footnote_numbers) + 1)


@dataclass(frozen=True, slots=True)
class _Cell:
    text: str
    colspan: int
    rowspan: int
    is_th: bool
    is_bold: bool


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""  # comments/PIs have non-str tags


def _attr(parent: ET.Element, child: str, name: str) -> str | None:
    el = parent.find(NS + child)
    return el.get(name) if el is not None else None


def _last_char(out: list[str]) -> str:
    return next((piece[-1] for piece in reversed(out) if piece), "")


def _drop_layout_gap(out: list[str]) -> None:
    """Remove source indentation (whitespace holding a line break) between a word and the <sub> after it."""
    j, gap = len(out), ""
    while j and not out[j - 1].strip():
        j -= 1
        gap = out[j] + gap
    if not j:
        return
    stripped = out[j - 1].rstrip()
    gap = out[j - 1][len(stripped) :] + gap
    if "\n" in gap and stripped[-1:].isalnum():
        out[j - 1] = stripped
        del out[j:]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in values:
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            out.append(value)
    return out


# ---- inline text ----


def _inline(el: ET.Element, ctx: _Context, out: list[str], skip: frozenset[str] = frozenset()) -> None:
    """Append el's text in document order; children named in `skip` are left to the caller (their tail is kept)."""
    if el.text:
        out.append(el.text)
    for child in el:
        name = _local(child.tag)
        if name in skip:
            pass
        elif name == "br":
            out.append("\n")
        elif name in ("sup", "sub"):
            inner = "".join(child.itertext())
            if name == "sub":
                _drop_layout_gap(out)
            if name == "sup" and NUMBER.fullmatch(inner.strip()) and _last_char(out).isdigit():
                out.append("^" + inner.strip())
            else:
                out.append(inner)
            if WORD_START.match(child.tail or ""):  # source omits the space: "B<sub>12</sub>levels"
                out.append(" ")
        elif name == "footnote":
            number = ctx.footnote_number(child.get("ID"))
            out.append(f" (footnote {number}) ")
            note: list[str] = []
            _inline(child, ctx, note)
            ctx.footnotes.append(f"(footnote {number}) {ctx.norm(''.join(note))}")
        elif name == "footnoteRef":
            out.append(f" (footnote {ctx.footnote_number(child.get('IDREF'))}) ")
        elif name == "renderMultiMedia":
            ctx.figures.append(child)
        elif name == "list":
            out.append("\n" + "\n".join(item.text for item in _list_items(child, ctx, 0)) + "\n")
        elif name == "table":
            out.append("\n" + _table_text(_table(child, ctx)) + "\n")
        elif name in NESTED_BLOCKS:
            out.append("\n")
            _inline(child, ctx, out)
            out.append("\n")
        else:
            if name not in INLINE:
                ctx.unknown[name] += 1
            _inline(child, ctx, out)
        if child.tail:
            out.append(child.tail)


def _text(el: ET.Element, ctx: _Context, *, keep_newlines: bool = False) -> str:
    out: list[str] = []
    _inline(el, ctx, out)
    return ctx.norm("".join(out), keep_newlines=keep_newlines)


# ---- lists and tables ----


def _list_items(lst: ET.Element, ctx: _Context, level: int) -> list[ListItem]:
    items: list[ListItem] = []
    for item in lst:
        name = _local(item.tag)
        if name == "caption":
            continue  # emitted as a CAPTION block by the caller
        if name != "item":
            ctx.unknown[f"list/{name}"] += 1
        out: list[str] = []
        _inline(item, ctx, out, skip=frozenset({"list"}))
        items.append(ListItem(ctx.norm("".join(out), keep_newlines=True), level))
        for nested in item.findall(NS + "list"):
            items.extend(_list_items(nested, ctx, level + 1))
    return [i for i in items if i.text]


def _span(value: str | None) -> int:
    try:
        return min(max(int(value or 1), 1), MAX_SPAN)
    except ValueError:
        return 1


def _all_bold(el: ET.Element, bold: bool = False) -> bool:
    bold = bold or "bold" in (el.get("styleCode") or "").lower()
    if (el.text or "").strip() and not bold:
        return False
    return all(_all_bold(c, bold) and (bold or not (c.tail or "").strip()) for c in el)


def _row(tr: ET.Element, ctx: _Context) -> list[_Cell]:
    cells = []
    for cell in tr:
        name = _local(cell.tag)
        if name not in ("td", "th"):
            ctx.unknown[f"tr/{name}"] += 1
            continue
        text = _text(cell, ctx, keep_newlines=True)
        cells.append(
            _Cell(
                text,
                _span(cell.get("colspan")),
                _span(cell.get("rowspan")),
                name == "th",
                bool(text) and _all_bold(cell),
            )
        )
    return cells


def _expand(rows: list[list[_Cell]]) -> tuple[list[tuple[str, ...]], bool]:
    """Rectangular grid with colspan/rowspan text repeated into every covered cell."""
    grid: list[list[str]] = []
    carry: dict[int, tuple[str, int]] = {}  # column -> (text, rows still covered)
    spans = False
    for row in rows:
        out: list[str] = []
        cells, col = list(row), 0
        while cells or any(c >= col for c in carry):
            if col in carry:
                text, left = carry.pop(col)
                if left > 1:
                    carry[col] = (text, left - 1)
                out.append(text)
                col += 1
            elif cells:
                cell = cells.pop(0)
                spans |= cell.colspan > 1 or cell.rowspan > 1
                for _ in range(cell.colspan):
                    if cell.rowspan > 1:
                        carry[col] = (cell.text, cell.rowspan - 1)
                    out.append(cell.text)
                    col += 1
            else:
                out.append("")
                col += 1
        grid.append(out)
    width = max((len(r) for r in grid), default=0)
    return [tuple(r + [""] * (width - len(r))) for r in grid], spans


def _table(table_el: ET.Element, ctx: _Context) -> Table:
    saved, ctx.footnotes = ctx.footnotes, []
    caption_el = table_el.find(NS + "caption")
    caption = _text(caption_el, ctx) if caption_el is not None else ""

    def trs(part: str) -> list[ET.Element]:
        return [tr for p in table_el.findall(NS + part) for tr in p.findall(NS + "tr")]

    head, body = (
        [_row(tr, ctx) for tr in trs("thead")],
        [_row(tr, ctx) for tr in trs("tbody") + table_el.findall(NS + "tr")],
    )
    foot = [" ".join(c.text.replace("\n", " ") for c in _row(tr, ctx) if c.text) for tr in trs("tfoot")]
    grid, spans = _expand(head + body)
    if head:
        n, source = len(head), HeaderSource.THEAD
    else:
        n = next((i for i, r in enumerate(body) if not (r and all(c.is_th for c in r))), len(body))
        source = HeaderSource.TH_CELLS
        if n == 0:
            first = [c for c in body[0] if c.text] if body else []
            n, source = (1, HeaderSource.STYLED) if first and all(c.is_bold for c in first) else (0, HeaderSource.NONE)
    if header_leaves_no_data(n, grid):
        n, source = 0, HeaderSource.NONE
    if source is HeaderSource.NONE and (n := infer_header_rows(grid)):
        source = HeaderSource.INFERRED
    layout = source is HeaderSource.NONE and is_layout(grid)
    footnotes = tuple(f for f in foot if f) + tuple(ctx.footnotes)
    ctx.footnotes = saved
    return Table(caption, tuple(grid[:n]), tuple(grid[n:]), source, spans, footnotes, layout)


def _table_text(table: Table) -> str:
    """Plain reading text for coverage and display; the embedding serialization is pending (P2-3)."""
    rows = (" | ".join(cell.replace("\n", " ") for cell in r) for r in (*table.header_rows, *table.body_rows))
    lines = [table.caption, *rows, *table.footnotes]
    return "\n".join(line for line in lines if line.strip(" |"))


# ---- blocks and sections ----


def _figure(media: ET.Element, ctx: _Context, pointer: str) -> list[Block]:
    ref = media.get("referencedObject") or ""
    om = ctx.media.get(ref)
    alt_el = om.find(NS + "text") if om is not None else None
    blocks = [Block(BlockKind.FIGURE, _text(alt_el, ctx) if alt_el is not None else "", f"{pointer}@{ref}")]
    caption_el = media.find(NS + "caption")
    if caption_el is not None and (caption := _text(caption_el, ctx)):
        blocks.append(Block(BlockKind.CAPTION, caption, f"{pointer}@{ref}/caption"))
    return blocks


def _blocks(container: ET.Element, ctx: _Context, pointer: str) -> list[Block]:
    blocks: list[Block] = []
    counts: Counter[str] = Counter()

    def loose(text: str | None) -> None:
        if text and (t := ctx.norm(text)):
            blocks.append(Block(BlockKind.PARAGRAPH, t, f"{pointer}/text()"))

    loose(container.text)
    for child in container:
        name = _local(child.tag)
        counts[name] += 1
        ptr = f"{pointer}/{name}[{counts[name]}]"
        ctx.footnotes, ctx.figures = [], []
        if name == "list":
            caption_el = child.find(NS + "caption")
            if caption_el is not None and (caption := _text(caption_el, ctx)):
                blocks.append(Block(BlockKind.CAPTION, caption, f"{ptr}/caption"))
            items = tuple(_list_items(child, ctx, 0))
            if items:
                blocks.append(Block(BlockKind.LIST, "\n".join(i.text for i in items), ptr, items=items))
        elif name == "table":
            table = _table(child, ctx)
            blocks.append(Block(BlockKind.TABLE, _table_text(table), ptr, table=table))
        elif name == "renderMultiMedia":
            blocks.extend(_figure(child, ctx, ptr))
        elif name != "br":
            if name not in INLINE and name != "paragraph":
                ctx.unknown[name] += 1
            if text := _text(child, ctx):
                blocks.append(Block(BlockKind.PARAGRAPH, text, ptr))
        blocks.extend(
            Block(BlockKind.PARAGRAPH, note, f"{ptr}/footnote[{k}]") for k, note in enumerate(ctx.footnotes, 1)
        )
        for k, media in enumerate(ctx.figures, 1):
            blocks.extend(_figure(media, ctx, f"{ptr}/renderMultiMedia[{k}]"))
        loose(child.tail)
    ctx.footnotes, ctx.figures = [], []
    return blocks


def _has_title(sec: ET.Element) -> bool:
    title = sec.find(NS + "title")
    return title is not None and bool("".join(title.itertext()).strip())


def _section(
    sec: ET.Element, ctx: _Context, path: str, parent_title: str | None, part: int, parent_role: SectionRole
) -> Section:
    code_el = sec.find(NS + "code")
    code = code_el.get("code") if code_el is not None else None
    section_id = _attr(sec, "id", "root") or f"path:{path}"

    ctx.footnotes, ctx.figures = [], []
    title_el = sec.find(NS + "title")
    printed = _text(title_el, ctx) if title_el is not None else ""
    title_notes = [
        Block(BlockKind.PARAGRAPH, n, f"section:{section_id}/title/footnote[{k}]")
        for k, n in enumerate(ctx.footnotes, 1)
    ]
    number = raw_number = None
    title = printed
    if m := SECTION_NUMBER.match(printed):
        raw_number, title = m[1], m[2]
        number = re.sub(r"\s+", "", raw_number)
    if title:
        title_source = TitleSource.SOURCE
    else:
        base = parent_title or (code_el.get("displayName") if code_el is not None else None) or "Untitled section"
        title, title_source = f"{base} (part {part})", TitleSource.DERIVED
    role = ROLES.get(code or "", parent_role)

    children: list[Section] = []
    for k, hl in enumerate(sec.findall(f"{NS}excerpt/{NS}highlight/{NS}text"), 1):
        if hl_blocks := _blocks(hl, ctx, f"section:{section_id}/excerpt/highlight[{k}]/text"):
            children.append(
                Section(
                    f"{section_id}:highlights{'' if k == 1 else k}",
                    f"{title} (highlights)",
                    TitleSource.DERIVED,
                    role=SectionRole.HIGHLIGHTS,
                    code=code,
                    blocks=hl_blocks,
                )
            )
    text_el = sec.find(NS + "text")
    blocks = title_notes + (_blocks(text_el, ctx, f"section:{section_id}/text") if text_el is not None else [])

    untitled = 0
    for i, comp in enumerate(sec.findall(NS + "component"), 1):
        child = comp.find(NS + "section")
        if child is None:
            continue
        untitled += not _has_title(child)
        children.append(_section(child, ctx, f"{path}.{i}", title, untitled, role))

    return Section(
        section_id,
        title,
        title_source,
        role=role,
        raw_title=printed if printed and printed != title else None,
        number=number,
        raw_number=raw_number,
        code=code,
        blocks=blocks,
        children=children,
    )


def _products(root: ET.Element, ctx: _Context) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {k: [] for k in ("brand", "generic", "ndc_product", "ndc_package", "form")}
    for product in root.iter(NS + "manufacturedProduct"):
        if (name := product.find(NS + "name")) is not None:
            found["brand"].append(_text(name, ctx))
        for generic in product.iterfind(f"{NS}asEntityWithGeneric/{NS}genericMedicine/{NS}name"):
            found["generic"].append(_text(generic, ctx))
        code = product.find(NS + "code")
        if code is not None and code.get("codeSystem") == NDC_SYSTEM:
            found["ndc_product"].append(code.get("code") or "")
        form = product.find(NS + "formCode")
        if form is not None:
            found["form"].append(form.get("displayName") or "")
    for package in root.iter(NS + "containerPackagedProduct"):
        code = package.find(NS + "code")
        if code is not None and code.get("codeSystem") == NDC_SYSTEM:
            found["ndc_package"].append(code.get("code") or "")
    return {k: _unique(v) for k, v in found.items()}


def parse_spl(
    xml: bytes, *, raw_file: str, raw_member: str | None, raw_sha256: str, alternate_files: tuple[str, ...] = ()
) -> ParsedDocument:
    root = ET.fromstring(xml)
    if root.tag != NS + "document":
        raise ValueError(f"not an SPL document: root element {root.tag!r}")
    set_id = _attr(root, "setId", "root")
    if not set_id:
        raise ValueError("SPL document without setId")
    ctx = _Context({om.get("ID") or "": om for om in root.iter(NS + "observationMedia")})
    products = _products(root, ctx)
    title_el = root.find(NS + "title")
    printed_title = _text(title_el, ctx) if title_el is not None else ""
    names = tuple(_unique(products["brand"] + products["generic"]))
    title = HIGHLIGHTS_TITLE.split(printed_title, maxsplit=1)[0].strip()  # drop the Highlights disclaimer
    if len(title) < 3 or NOT_A_TITLE.match(title):
        title = names[0] if names else f"DailyMed label {set_id}"

    sections: list[Section] = []
    body = root.find(f"{NS}component/{NS}structuredBody")
    untitled = 0
    for i, comp in enumerate(body.findall(NS + "component") if body is not None else [], 1):
        sec = comp.find(NS + "section")
        if sec is not None:
            untitled += not _has_title(sec)
            sections.append(_section(sec, ctx, str(i), None, untitled, SectionRole.BODY))

    metadata: dict[str, str | list[str]] = {
        "spl_id": _attr(root, "id", "root") or "",
        "document_type": _attr(root, "code", "displayName") or "",
        "effective_time": _attr(root, "effectiveTime", "value") or "",
        "spl_title": printed_title,
        "brand_names": products["brand"],
        "generic_names": products["generic"],
        "ndc_product_codes": products["ndc_product"],
        "ndc_package_codes": products["ndc_package"],
        "dosage_forms": products["form"],
        "unknown_elements": [f"{k}:{v}" for k, v in sorted(ctx.unknown.items())],
    }
    return ParsedDocument(
        document_id=f"dailymed:{set_id}",
        source="dailymed",
        source_format="spl_xml",
        document_version=_attr(root, "versionNumber", "value") or "",
        title=title,
        medicine_names=names,
        source_reference=SOURCE_URL.format(set_id),
        raw_file=raw_file,
        raw_member=raw_member,
        raw_sha256=raw_sha256,
        alternate_files=alternate_files,
        metadata=metadata,
        sections=sections,
        normalizer_version=NORMALIZER_VERSION,
        normalization_changes=dict(ctx.changes),
    )
