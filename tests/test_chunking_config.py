from pathlib import Path

import pytest
from pydantic import ValidationError

from medical_rag.chunking import config as config_module
from medical_rag.chunking.config import ChunkingConfig, TokenizerSpec
from medical_rag.ingestion.models import SectionRole

OPEN = {
    "overlap_tokens": 0,
    "overlap_method": "sentence",
    "overlap_fit": "reserve",
    "highlights": "exclude",
    "abstracts": "include",
    "table_format": "pipe",
}


def make(**changes):
    return ChunkingConfig(**{**OPEN, **changes})


def test_open_decisions_have_no_default():
    with pytest.raises(ValidationError) as err:
        ChunkingConfig()
    missing = {e["loc"][0] for e in err.value.errors() if e["type"] == "missing"}
    assert missing == {"overlap_tokens", "overlap_method", "overlap_fit", "highlights", "abstracts", "table_format"}


@pytest.mark.parametrize("overlap", [0, 32, 64, 96])
def test_overlap_candidates_accepted(overlap):
    assert make(overlap_tokens=overlap).overlap_tokens == overlap


def test_other_overlap_rejected():
    with pytest.raises(ValidationError):
        make(overlap_tokens=48)


def test_budget_is_512_for_current_experiment():
    cfg = make()
    assert (cfg.max_tokens, cfg.reserved_special_tokens, cfg.budget) == (512, 0, 512)


def test_config_id_deterministic_and_16_hex():
    assert make().config_id == make().config_id
    assert len(make().config_id) == 16
    int(make().config_id, 16)


@pytest.mark.parametrize(
    "changes",
    [
        {"overlap_tokens": 32},
        {"overlap_method": "word"},
        {"overlap_fit": "fill"},
        {"highlights": "include"},
        {"abstracts": "exclude"},
        {"table_format": "key_value"},
        {"max_tokens": 256},
        {"tokenizer": TokenizerSpec(revision="other")},
        {"tokenizer": TokenizerSpec(sha256="0" * 64)},
    ],
)
def test_config_id_changes_with_settings(changes):
    assert make(**changes).config_id != make().config_id


def test_config_id_changes_with_chunker_version(monkeypatch):
    before = make().config_id
    monkeypatch.setattr(config_module, "CHUNKER_VERSION", "999")
    assert make().config_id != before


def test_config_id_ignores_tokenizer_path():
    moved = make(tokenizer=TokenizerSpec(file=Path("elsewhere/tokenizer.json")))
    assert moved.config_id == make().config_id


@pytest.mark.parametrize(
    "role", [SectionRole.BOXED_WARNING, SectionRole.CASE_REPORT, SectionRole.HIGHLIGHTS, SectionRole.ABSTRACT]
)
def test_approved_or_open_roles_cannot_be_excluded(role):
    with pytest.raises(ValidationError):
        make(excluded_roles=(SectionRole.PACKAGE_LABEL, role))


def test_default_exclusions_are_the_approved_ones():
    assert make().excluded_roles == (SectionRole.PACKAGE_LABEL, SectionRole.PRODUCT_DATA)
