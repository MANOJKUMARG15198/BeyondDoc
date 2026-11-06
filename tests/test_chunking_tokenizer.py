"""Real BERT-uncased tests; skipped when the local tokenizer file (gitignored data/) is missing, e.g. in CI."""

import random

import pytest

from medical_rag.chunking.config import TokenizerSpec
from medical_rag.chunking.tokenizer import ChunkingError, TokenCounter

SPEC = TokenizerSpec()
pytestmark = pytest.mark.skipif(not SPEC.file.exists(), reason="local BERT tokenizer file not available")
CHEMICAL = "4-[4-(4-chlorophenyl)-4-hydroxypiperidin-1-yl]-1-(4-fluorophenyl)butan-1-one"


@pytest.fixture(scope="module")
def counter():
    return TokenCounter(SPEC)


def assert_complete(counter, text, spans, limit):
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))  # no gaps, no overlaps
    assert "".join(text[s:e] for s, e, _ in spans) == text
    assert all(counter.count(text[s:e]) <= limit for s, e, _ in spans)


def test_no_special_tokens(counter):
    assert counter.count("") == 0
    assert counter.count("a") == 1


def test_known_counts(counter):
    assert counter.count("atorvastatin") == 5  # matches tokenizer_eval_batch_01


def test_counting_never_truncates(counter):
    assert counter.count("dose " * 2000) == 2000


def test_wrong_tokenizer_file_rejected(tmp_path):
    fake = tmp_path / "tokenizer.json"
    fake.write_text("{}", encoding="utf-8")
    with pytest.raises(ChunkingError):
        TokenCounter(TokenizerSpec(file=fake))


def test_pharmaceutical_name_kept_whole_when_it_fits(counter):
    assert counter.split_long_word("atorvastatin", 5) == [(0, 12, "none")]


def test_chemical_name_cut_at_punctuation(counter):
    spans = counter.split_long_word(CHEMICAL, 8)
    assert_complete(counter, CHEMICAL, spans, 8)
    assert {kind for *_, kind in spans[:-1]} == {"punctuation"}


def test_uninterrupted_string_uses_reported_subword_cuts(counter):
    text = "atorvastatinatorvastatinatorvastatin"  # one word, no punctuation
    spans = counter.split_long_word(text, 4)
    assert_complete(counter, text, spans, 4)
    assert "subword" in {kind for *_, kind in spans}


def test_unicode_offsets_rejoin_exactly(counter):
    text = "≥10µg/m²±0.5💊…" * 50
    spans = counter.split_long_word(text, 6)
    assert_complete(counter, text, spans, 6)


def test_every_cut_is_verified(counter):
    rng = random.Random(20261104)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-/()[]µ²≥"
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80)))
        limit = rng.randint(1, 20)
        assert_complete(counter, text, counter.split_long_word(text, limit), limit)


def test_invalid_limit_raises(counter):
    with pytest.raises(ChunkingError):
        counter.safe_cut("atorvastatin", 0)
