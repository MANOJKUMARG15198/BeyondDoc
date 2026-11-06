"""PMC JATS XML parser: one article -> ParsedDocument (pure: bytes in, no file access).

Content is tagged with a SectionRole and never excluded here. Abstracts are ABSTRACT sections (P2-7 decides
their use at chunking); declarations, acknowledgements, footnote groups, notes and appendices are BACK_MATTER
(P2-19, chunked); the reference list goes to metadata only (P2-6). Floating tables and figures (floats-group)
are placed after the block that first cites them (PMC-1); any never cited go to one derived section at the end.
A correction notice becomes a RelatedDocument, so chunks carry "corrected_article". Author names are kept as
metadata, e-mail addresses never (J-2). The block builder is shared with the LiverTox NXML parser.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable

from medical_rag.ingestion.models import (
    Block,
    BlockKind,
    EvidenceType,
    HeaderSource,
    ListItem,
    ParsedDocument,
    RelatedDocument,
    Section,
    SectionRole,
    Table,
    TitleSource,
)
from medical_rag.ingestion.spl import SECTION_NUMBER, _Cell, _expand, _span
from medical_rag.ingestion.table_heuristics import header_leaves_no_data, infer_header_rows, is_layout
from medical_rag.ingestion.text import NORMALIZER_VERSION
from medical_rag.ingestion.xml_text import XLINK, InlineContext, inline, inline_child, local, text_of

SOURCE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/{}/"
BACK_MATTER_TITLE = re.compile(
    r"^(acknowledg|funding|financial (support|disclosure)|conflicts? of interest|competing interests?|disclosures?|"
    r"author(s'|s)? contributions?|contributions|ethic|data availability|informed consent|declarations?|"
    r"consent for publication|abbreviations)",
    re.IGNORECASE,
)
PARAGRAPH_BLOCKS = {"list", "table-wrap", "fig", "disp-formula", "boxed-text", "disp-quote", "def-list", "statement"}
CONTAINERS = {"boxed-text", "disp-quote", "statement", "fn-group", "fn", "app", "notes", "glossary", "sec-meta"}
SKIPPED = {"title", "label", "sec", "ref-list", "object-id", "alternatives"}


class Builder:
    """Turns JATS-family section content into Blocks; tracks floats, footnote labels and inline counts."""

    def __init__(self, floats: dict[str, ET.Element] | None = None, *, references_as_blocks: bool = False) -> None:
        self.ctx = InlineContext()
        self.floats = floats or {}
        self.placed: set[str] = set()
        self.unknown: Counter[str] = Counter()
        self.references_as_blocks = references_as_blocks

    # ---- sections ----

    def section(
        self,
        sec: ET.Element,
        path: str,
        role: SectionRole,
        parent_title: str | None = None,
        *,
        case_rule: Callable[[str], bool] | None = None,
        case_id: str | None = None,
        evidence: EvidenceType = EvidenceType.GENERAL,
    ) -> Section:
        section_id = sec.get("id") or f"path:{path}"
        printed = text_of(sec.find("title"), self.ctx)
        label = text_of(sec.find("label"), self.ctx)
        number = raw_number = None
        title = printed
        if label:
            raw_number, number = label, label.rstrip(".")
        elif m := SECTION_NUMBER.match(printed):
            raw_number, title = m[1], m[2]
            number = re.sub(r"\s+", "", raw_number)
        title_source = TitleSource.SOURCE if title else TitleSource.DERIVED
        title = title or f"{parent_title or 'Untitled section'} (part {path.rsplit('.', 1)[-1]})"
        if role is SectionRole.BODY and BACK_MATTER_TITLE.match(title):
            role = SectionRole.BACK_MATTER
        if case_id is None and case_rule is not None and case_rule(title):
            role, case_id, evidence = SectionRole.CASE_REPORT, section_id, EvidenceType.INDIVIDUAL_CASE
        blocks = self.blocks(sec, f"section:{section_id}", role)
        children = [
            self.section(c, f"{path}.{i}", role, title, case_rule=case_rule, case_id=case_id, evidence=evidence)
            for i, c in enumerate(sec.findall("sec"), 1)
        ]
        return Section(
            section_id,
            title,
            title_source,
            role=role,
            raw_title=printed if printed and printed != title else None,
            number=number,
            raw_number=raw_number,
            evidence_type=evidence,
            case_id=case_id,
            blocks=blocks,
            children=children,
        )

    # ---- blocks ----

    def blocks(
        self,
        container: ET.Element,
        pointer: str,
        role: SectionRole = SectionRole.BODY,
        skip: frozenset[str] = frozenset(),
    ) -> list[Block]:
        out: list[Block] = []
        counts: Counter[str] = Counter()
        for child in container:
            name = local(child.tag)
            if name in skip:
                continue
            if name == "ref-list" and self.references_as_blocks:
                counts[name] += 1
                ptr = f"{pointer}/ref-list[{counts[name]}]"
                out += [
                    Block(BlockKind.REFERENCE, t, f"{ptr}/ref[{k}]")
                    for k, ref in enumerate(child.findall("ref"), 1)
                    if (t := text_of(ref, self.ctx))
                ]
                continue
            if name in SKIPPED or not name:
                continue
            counts[name] += 1
            self.ctx.references = []
            out += self.element(child, f"{pointer}/{name}[{counts[name]}]", role)
            out += self._place_floats()
        return out

    def element(self, el: ET.Element, ptr: str, role: SectionRole) -> list[Block]:
        name = local(el.tag)
        if name == "p":
            return self.paragraph(el, ptr, role)
        if name == "list":
            return self.list_block(el, ptr)
        if name == "table-wrap":
            return self.table_wrap(el, ptr)
        if name == "fig":
            return self.figure(el, ptr)
        if name == "def-list":
            pairs = (
                (text_of(d.find("term"), self.ctx), text_of(d.find("def"), self.ctx)) for d in el.findall("def-item")
            )
            items = tuple(ListItem(f"{term}: {value}" if term else value) for term, value in pairs if term or value)
            title = text_of(el.find("title"), self.ctx)
            lead = [Block(BlockKind.CAPTION, title, f"{ptr}/title")] if title else []
            return lead + ([Block(BlockKind.LIST, "\n".join(i.text for i in items), ptr, items=items)] if items else [])
        if name in CONTAINERS:
            heading = text_of(el.find("caption"), self.ctx) or text_of(el.find("title"), self.ctx)
            label = text_of(el.find("label"), self.ctx)
            lead = [Block(BlockKind.CAPTION, f"{label} {heading}".strip(), f"{ptr}/caption")] if heading else []
            inner = self.blocks(el, ptr, role, skip=frozenset({"caption"}))
            nested = [
                b for i, s in enumerate(el.findall("sec"), 1) for b in self._flat_section(s, f"{ptr}/sec[{i}]", role)
            ]
            if name == "fn" and label and inner and inner[0].kind is BlockKind.PARAGRAPH:
                inner[0] = Block(BlockKind.PARAGRAPH, f"{label} {inner[0].text}", inner[0].source_pointer)
            return lead + inner + nested
        if name in ("caption", "disp-formula"):
            if name == "disp-formula":
                self.ctx.formulas += 1
            text = text_of(el, self.ctx)
            return [Block(BlockKind.CAPTION if name == "caption" else BlockKind.PARAGRAPH, text, ptr)] if text else []
        self.unknown[name] += 1
        text = text_of(el, self.ctx)
        return [Block(BlockKind.PARAGRAPH, text, ptr)] if text else []

    def _flat_section(self, sec: ET.Element, ptr: str, role: SectionRole) -> list[Block]:
        """A <sec> nested in a box or appendix: its title as a caption, then its content."""
        title = text_of(sec.find("title"), self.ctx)
        blocks = [Block(BlockKind.CAPTION, title, f"{ptr}/title")] if title else []
        blocks += self.blocks(sec, ptr, role)
        return blocks + [
            b for i, s in enumerate(sec.findall("sec"), 1) for b in self._flat_section(s, f"{ptr}/sec[{i}]", role)
        ]

    def paragraph(self, p: ET.Element, ptr: str, role: SectionRole) -> list[Block]:
        """A <p>, split around any block element nested inside it (lists, tables, figures, boxes)."""
        blocks: list[Block] = []
        buffer: list[str] = [p.text] if p.text else []
        parts, nested = 0, Counter()

        def flush() -> None:
            nonlocal parts
            if text := self.ctx.norm("".join(buffer)):
                parts += 1
                blocks.append(Block(BlockKind.PARAGRAPH, text, ptr if parts == 1 else f"{ptr}/text[{parts}]"))
            buffer.clear()

        for child in p:
            name = local(child.tag)
            if name in PARAGRAPH_BLOCKS:
                flush()
                nested[name] += 1
                blocks += self.element(child, f"{ptr}/{name}[{nested[name]}]", role)
                if child.tail:
                    buffer.append(child.tail)
            else:
                inline_child(child, self.ctx, buffer)
        flush()
        return blocks

    def list_block(self, lst: ET.Element, ptr: str) -> list[Block]:
        caption = text_of(lst.find("title"), self.ctx)
        blocks = [Block(BlockKind.CAPTION, caption, f"{ptr}/caption")] if caption else []
        items = tuple(self._items(lst, 0))
        if items:
            blocks.append(Block(BlockKind.LIST, "\n".join(i.text for i in items), ptr, items=items))
        return blocks

    def _items(self, lst: ET.Element, level: int) -> list[ListItem]:
        items: list[ListItem] = []
        for item in lst.findall("list-item"):
            out: list[str] = []
            inline(item, self.ctx, out, skip=frozenset({"list", "label"}))
            text = self.ctx.norm("".join(out), keep_newlines=True)
            if label := text_of(item.find("label"), self.ctx):
                text = f"{label} {text}".strip()
            if text:
                items.append(ListItem(text, level))
            for nested in item.findall("list"):
                items += self._items(nested, level + 1)
        return items

    def figure(self, fig: ET.Element, ptr: str) -> list[Block]:
        graphic = fig.find(".//graphic")
        href = graphic.get(f"{XLINK}href", "") if graphic is not None else ""
        pointer = f"{ptr}@{href}" if href else ptr
        alt = text_of(fig.find(".//alt-text"), self.ctx)
        label = text_of(fig.find("label"), self.ctx)
        caption = f"{label} {text_of(fig.find('caption'), self.ctx)}".strip()
        blocks = [Block(BlockKind.FIGURE, alt, pointer)]
        if caption:
            blocks.append(Block(BlockKind.CAPTION, caption, f"{pointer}/caption"))
        return blocks

    def table_wrap(self, wrap: ET.Element, ptr: str) -> list[Block]:
        label = text_of(wrap.find("label"), self.ctx)
        caption = f"{label} {text_of(wrap.find('caption'), self.ctx)}".strip()
        tables = wrap.findall(".//table")
        if not tables:  # an image of a table: keep what text there is
            return self.figure(wrap, ptr)
        footnotes: list[str] = []
        for foot in wrap.findall("table-wrap-foot"):
            for child in foot:  # general notes: anything that is not a footnote
                if local(child.tag) not in ("fn", "fn-group") and (note := text_of(child, self.ctx)):
                    footnotes.append(note)
            for k, fn in enumerate(foot.iter("fn"), 1):
                mark = text_of(fn.find("label"), self.ctx) or fn.get("symbol") or str(k)
                out: list[str] = []
                inline(fn, self.ctx, out, skip=frozenset({"label"}))
                if note := self.ctx.norm("".join(out)):
                    footnotes.append(f"(footnote {mark}) {note}")
        if len(tables) == 1:
            return [self._table(tables[0], caption, footnotes, ptr)]
        # several <table> elements in one wrap: one block each, sharing the caption and the wrap's notes
        return [
            self._table(t, f"{caption} (table {k} of {len(tables)})".strip(), footnotes, f"{ptr}/table[{k}]")
            for k, t in enumerate(tables, 1)
        ]

    def _table(self, table_el: ET.Element, caption: str, footnotes: list[str], ptr: str) -> Block:
        def rows(part: str) -> list[ET.Element]:
            return [tr for p in table_el.findall(part) for tr in p.findall("tr")]

        head = [self._row(tr) for tr in rows("thead")]
        body = [self._row(tr) for tr in rows("tbody") + table_el.findall("tr")]
        foot_rows = [" ".join(c.text for c in self._row(tr) if c.text) for tr in rows("tfoot")]
        grid, spans = _expand(head + body)
        if head:
            n, source = len(head), HeaderSource.THEAD
        else:
            n = next((i for i, r in enumerate(body) if not (r and all(c.is_th for c in r))), len(body))
            source = HeaderSource.TH_CELLS
            if n == 0:
                first = [c for c in body[0] if c.text] if body else []
                n, source = (
                    (1, HeaderSource.STYLED) if first and all(c.is_bold for c in first) else (0, HeaderSource.NONE)
                )
        if header_leaves_no_data(n, grid):
            n, source = 0, HeaderSource.NONE
        if source is HeaderSource.NONE and (n := infer_header_rows(grid)):
            source = HeaderSource.INFERRED
        layout = source is HeaderSource.NONE and is_layout(grid)
        table = Table(caption, tuple(grid[:n]), tuple(grid[n:]), source, spans, tuple(foot_rows + footnotes), layout)
        return Block(BlockKind.TABLE, _reading_text(table), ptr, table=table)

    def _row(self, tr: ET.Element) -> list[_Cell]:
        cells = []
        for cell in tr:
            name = local(cell.tag)
            if name not in ("td", "th"):
                continue
            text = text_of(cell, self.ctx, keep_newlines=True)
            bold = bool(text) and _all_bold(cell)
            cells.append(_Cell(text, _span(cell.get("colspan")), _span(cell.get("rowspan")), name == "th", bold))
        return cells

    def _place_floats(self) -> list[Block]:
        blocks: list[Block] = []
        for rid in dict.fromkeys(self.ctx.references):
            el = self.floats.get(rid)
            if el is not None and rid not in self.placed:
                self.placed.add(rid)
                blocks += self.element(el, f"floats-group/{local(el.tag)}[@id='{rid}']", SectionRole.BODY)
        self.ctx.references = []
        return blocks

    def unplaced_floats(self) -> list[Block]:
        self.ctx.references = [rid for rid in self.floats if rid not in self.placed]
        return self._place_floats()


def _all_bold(el: ET.Element, bold: bool = False) -> bool:
    bold = bold or local(el.tag) == "bold"
    if (el.text or "").strip() and not bold:
        return False
    return all(_all_bold(c, bold) and (bold or not (c.tail or "").strip()) for c in el)


def _reading_text(table: Table) -> str:
    rows = (" | ".join(c.replace("\n", " ") for c in r) for r in (*table.header_rows, *table.body_rows))
    return "\n".join(line for line in (table.caption, *rows, *table.footnotes) if line.strip(" |"))


def iso_date(el: ET.Element | None) -> str:
    """ISO date from JATS <year>/<month>/<day> children (partial dates allowed)."""
    if el is None:
        return ""
    if iso := el.get("iso-8601-date"):
        return iso[:10]
    year, month, day = ((el.findtext(t) or "").strip() for t in ("year", "month", "day"))
    if not year.isdigit():
        return ""
    out = [year]
    if month.isdigit():
        out.append(month.zfill(2))
        if day.isdigit():
            out.append(day.zfill(2))
    return "-".join(out)


def parse_jats(
    xml: bytes, *, raw_file: str, raw_member: str | None, raw_sha256: str, alternate_files: tuple[str, ...] = ()
) -> ParsedDocument:
    root = ET.fromstring(xml)
    if local(root.tag) != "article":
        raise ValueError(f"not a JATS article: root element {root.tag!r}")
    meta = root.find("front/article-meta")
    if meta is None:
        raise ValueError("JATS article without article-meta")
    ids = {a.get("pub-id-type"): (a.text or "").strip() for a in meta.findall("article-id")}
    pmcid = ids.get("pmcid") or ids.get("pmc") or ""
    if not pmcid:
        raise ValueError("JATS article without a PMC id")
    versioned = ids.get("pmcid-ver") or pmcid
    floats = {el.get("id"): el for el in root.findall("floats-group/*") if el.get("id")}
    b = Builder(floats)
    title = text_of(meta.find("title-group/article-title"), b.ctx)

    sections: list[Section] = []
    for k, abstract in enumerate(meta.findall("abstract"), 1):
        sid = abstract.get("id") or f"abstract[{k}]"
        heading = text_of(abstract.find("title"), b.ctx) or ("Abstract" if k == 1 else f"Abstract {k}")
        children = [
            b.section(s, f"abstract{k}.{i}", SectionRole.ABSTRACT, heading)
            for i, s in enumerate(abstract.findall("sec"), 1)
        ]
        sections.append(
            Section(
                sid,
                heading,
                TitleSource.SOURCE if abstract.find("title") is not None else TitleSource.DERIVED,
                role=SectionRole.ABSTRACT,
                blocks=b.blocks(abstract, f"section:{sid}", SectionRole.ABSTRACT),
                children=children,
            )
        )
    body = root.find("body")
    if body is not None:
        loose = [c for c in body if local(c.tag) not in ("sec", "")]
        if loose:
            holder = ET.Element("body")
            holder.extend(loose)
            heading = text_of(loose[0].find("caption/title"), b.ctx) or "Article text"
            sections.append(
                Section("path:body.0", heading, TitleSource.DERIVED, blocks=b.blocks(holder, "section:path:body.0"))
            )
        sections += [b.section(s, f"body.{i}", SectionRole.BODY) for i, s in enumerate(body.findall("sec"), 1)]
    back = root.find("back")
    references: list[str] = []
    if back is not None:
        counts: Counter[str] = Counter()
        for child in back:
            name = local(child.tag)
            counts[name] += 1
            path = f"back.{name}{counts[name]}"
            if name == "ref-list":
                references += [t for ref in child.iter("ref") if (t := text_of(ref, b.ctx))]
            elif name == "sec":
                sections.append(b.section(child, path, SectionRole.BACK_MATTER))
            elif name == "app-group":
                for i, app in enumerate(child.findall("app"), 1):
                    sections.append(b.section(app, f"{path}.{i}", SectionRole.BACK_MATTER))
            elif name:
                sid = child.get("id") or f"path:{path}"
                heading = text_of(child.find("title"), b.ctx) or {"ack": "Acknowledgments"}.get(name, "Notes")
                sections.append(
                    Section(
                        sid,
                        heading,
                        TitleSource.SOURCE if child.find("title") is not None else TitleSource.DERIVED,
                        role=SectionRole.BACK_MATTER,
                        blocks=b.blocks(child, f"section:{sid}", SectionRole.BACK_MATTER),
                        children=[
                            b.section(s, f"{path}.{i}", SectionRole.BACK_MATTER)
                            for i, s in enumerate(child.findall("sec"), 1)
                        ],
                    )
                )
    if leftovers := b.unplaced_floats():
        sections.append(Section("floats-unreferenced", "Tables and figures", TitleSource.DERIVED, blocks=leftovers))

    related = tuple(
        RelatedDocument(r.get("related-article-type") or "", r.get(f"{XLINK}href") or "", text_of(r, b.ctx), False)
        for r in root.iter("related-article")
    )
    authors = [
        " ".join(p for p in ((n.findtext("given-names") or "").strip(), (n.findtext("surname") or "").strip()) if p)
        for n in meta.findall("contrib-group/contrib/name")
    ]
    dates = {
        f"pub_date_{d.get('pub-type') or d.get('date-type') or 'unknown'}": iso_date(d)
        for d in meta.findall("pub-date")
    }
    license_el = meta.find("permissions/license")
    metadata: dict[str, str | list[str]] = {
        **{k: v for k, v in (("pmcid", pmcid), ("pmid", ids.get("pmid", "")), ("doi", ids.get("doi", ""))) if v},
        "journal": text_of(root.find("front/journal-meta/journal-title-group/journal-title"), b.ctx),
        "article_type": root.get("article-type") or "",
        **{k: v for k, v in dates.items() if v},
        "license": (license_el.get(f"{XLINK}href") or text_of(license_el, b.ctx)[:200])
        if license_el is not None
        else "",
        "keywords": [t for k in meta.iter("kwd") if (t := text_of(k, b.ctx))],
        "authors": [a for a in authors if a],
        "references": references,
        "formulas": str(b.ctx.formulas),
        "emails_removed": str(b.ctx.emails_removed),
        "unknown_elements": [f"{k}:{v}" for k, v in sorted(b.unknown.items())],
    }
    return ParsedDocument(
        document_id=f"pmc_oa:{versioned}",
        source="pmc_oa",
        source_format="jats_xml",
        document_version=versioned.rsplit(".", 1)[1] if "." in versioned else "",
        title=title or pmcid,
        medicine_names=(),
        source_reference=SOURCE_URL.format(pmcid),
        raw_file=raw_file,
        raw_member=raw_member,
        raw_sha256=raw_sha256,
        alternate_files=alternate_files,
        related_documents=related,
        metadata=metadata,
        sections=sections,
        normalizer_version=NORMALIZER_VERSION,
        normalization_changes=dict(b.ctx.changes),
    )
