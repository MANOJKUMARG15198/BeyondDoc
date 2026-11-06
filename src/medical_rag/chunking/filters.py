"""Content filtering before chunking (M7). Decisions only: the parse is never changed, every exclusion is recorded.

Section roles excluded by configuration (package labels, product data) never become chunks; their product
identifiers stay in the document metadata. Highlights and abstracts follow their open settings. Figures keep only
source text (alt text and caption, never a generated description); package-label images and figures without
meaningful text are excluded.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from medical_rag.chunking.config import ChunkingConfig
from medical_rag.ingestion.models import Section, SectionRole


class Decision(NamedTuple):
    keep: bool
    reason: str | None


KEEP = Decision(True, None)


def section_decision(section: Section, cfg: ChunkingConfig) -> Decision:
    if section.role in cfg.excluded_roles:
        return Decision(False, f"excluded_role:{section.role.value}")
    if section.role is SectionRole.HIGHLIGHTS and cfg.highlights != "include":
        return Decision(False, f"highlights:{cfg.highlights}")
    if section.role is SectionRole.ABSTRACT and cfg.abstracts != "include":
        return Decision(False, f"abstracts:{cfg.abstracts}")
    return KEEP  # boxed warnings, case reports, letter signatures, back matter: kept


def figure_decision(alt_text: str, caption: str, cfg: ChunkingConfig) -> Decision:
    text = f"{alt_text} {caption}"
    if any(re.search(p, text, re.IGNORECASE) for p in cfg.package_image_patterns):
        return Decision(False, "package_label_image")
    if not caption and len(alt_text.split()) < cfg.min_figure_words:
        return Decision(False, "no_meaningful_text")
    return KEEP
