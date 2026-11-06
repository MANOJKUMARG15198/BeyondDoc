"""Chunking configuration (ADR-010).

Every value is experimental unless docs/DECISION_LOG.md records it as decided. Open decisions have no default:
each experiment must state them, so a default can never become a decision by accident.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medical_rag.ingestion.models import SectionRole

CHUNKER_VERSION = "2"  # bump whenever chunking output can change for the same input and configuration
_TOKENIZER_FILE = Path(
    "data/models/tokenizers/google-bert__bert-base-uncased/86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json"
)
_ALWAYS_KEPT = (SectionRole.BOXED_WARNING, SectionRole.CASE_REPORT)  # approved rules


class TokenizerSpec(BaseModel):
    """Identity of the counting tokenizer. The file path says where to load it and is not part of its identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "google-bert/bert-base-uncased"
    revision: str = "86b5e0934494bd15c9632b12f734a8a67f723594"
    sha256: str = "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98"  # tokenizer.json
    file: Path = Field(default=_TOKENIZER_FILE, exclude=True)


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tokenizer: TokenizerSpec = TokenizerSpec()
    max_tokens: int = Field(512, ge=32)  # provisional ceiling for prefix + separator + chunk text
    reserved_special_tokens: int = Field(0, ge=0)  # 0 for the current experiment; not a fit guarantee for embedders
    prefix_template: str = "{title} — {path}"  # P2-2 provisional; counted inside the ceiling
    prefix_separator: str = "\n"
    excluded_roles: tuple[SectionRole, ...] = (SectionRole.PACKAGE_LABEL, SectionRole.PRODUCT_DATA)
    table_notes_max_share: float = Field(0.25, gt=0, lt=1)  # general table notes repeat while within this share
    min_figure_words: int = Field(3, ge=1)  # alt text shorter than this, without a caption, is not meaningful
    package_image_patterns: tuple[str, ...] = (
        r"\bNDC\b",
        r"principal display panel",
        r"\bcarton\b",
        r"container label",
        r"\bblister\b",
    )
    long_word_chars: int = 100  # BERT WordPiece turns a longer unbroken word into one [UNK]: flagged, not altered

    # Open experimental decisions: required, no defaults.
    overlap_tokens: Literal[0, 32, 64, 96]
    overlap_method: Literal["sentence", "word"]
    overlap_fit: Literal["reserve", "fill"]  # O-1: base chunks packed to budget - N ("reserve") or to budget ("fill")
    highlights: Literal["include", "exclude", "metadata_only"]  # H-1 open
    abstracts: Literal["include", "exclude", "metadata_only"]  # P2-7 open
    table_format: Literal["pipe", "key_value"]

    @model_validator(mode="after")
    def _approved_roles_stay_in(self) -> ChunkingConfig:
        own_setting = (SectionRole.HIGHLIGHTS, SectionRole.ABSTRACT)
        if blocked := [r.value for r in self.excluded_roles if r in _ALWAYS_KEPT or r in own_setting]:
            raise ValueError(f"roles cannot be excluded here: {blocked} (highlights/abstracts have their own setting)")
        return self

    @property
    def budget(self) -> int:
        return self.max_tokens - self.reserved_special_tokens

    @property
    def config_id(self) -> str:
        """Deterministic: canonical JSON of every setting (tokenizer by name, revision, checksum) + CHUNKER_VERSION."""
        payload = {"chunker_version": CHUNKER_VERSION, **self.model_dump(mode="json")}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
