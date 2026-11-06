import re

import pytest


class FakeCounter:
    """Deterministic stand-in for TokenCounter: one token per word or punctuation mark (like BERT's
    pre-tokenizer without WordPiece), so chunking tests run in CI without the local tokenizer file."""

    TOKEN = re.compile(r"\w+|[^\w\s]")

    def count(self, text):
        return len(self.TOKEN.findall(text))

    def split_long_word(self, text, limit):
        bounds = [m.start() for m in self.TOKEN.finditer(text)][limit::limit]
        edges = [0, *bounds, len(text)]
        return [(a, b, "word_start") for a, b in zip(edges, edges[1:], strict=False)]


@pytest.fixture
def fake_counter():
    return FakeCounter()


@pytest.fixture(autouse=True)
def no_real_tracing(monkeypatch):
    """Tests never send traces: a developer's .env may set MEDRAG_TRACING=langfuse with real keys."""
    from medical_rag.core.config import get_settings
    from medical_rag.observability import tracing

    monkeypatch.setenv("MEDRAG_TRACING", "none")
    get_settings.cache_clear()
    tracing._tracer_for.cache_clear()
    yield
    get_settings.cache_clear()
    tracing._tracer_for.cache_clear()
