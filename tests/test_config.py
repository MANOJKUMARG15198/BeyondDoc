import pytest
from pydantic import ValidationError

from medical_rag.core.config import Settings


def test_defaults_are_local_and_do_not_log_query_text():
    s = Settings(_env_file=None)
    assert s.env == "local"
    assert s.log_query_text is False
    assert str(s.opensearch_url) == "http://127.0.0.1:9200/"
    assert s.anthropic_api_key is None or "SecretStr" in repr(s.anthropic_api_key)  # never printed in plain text


def test_env_prefix_overrides(monkeypatch):
    monkeypatch.setenv("MEDRAG_OPENSEARCH_URL", "http://example.test:9201")
    monkeypatch.setenv("MEDRAG_LOG_LEVEL", "DEBUG")
    s = Settings(_env_file=None)
    assert s.opensearch_url.host == "example.test"
    assert s.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("name", "value"),
    [("MEDRAG_LOG_LEVEL", "LOUD"), ("MEDRAG_OPENSEARCH_TIMEOUT_S", "0"), ("MEDRAG_OPENSEARCH_URL", "not a url")],
)
def test_invalid_values_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
