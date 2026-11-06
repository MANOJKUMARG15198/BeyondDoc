from fastapi.testclient import TestClient
from opensearchpy.exceptions import ConnectionError as OSConnectionError

from medical_rag.api.main import app, get_search_client


class FakeCluster:
    def __init__(self, status="green", fail=False):
        self._status, self._fail = status, fail

    def health(self):
        if self._fail:
            raise OSConnectionError("N/A", "refused", None)
        return {"status": self._status, "number_of_nodes": 1}


class FakeClient:
    def __init__(self, **kw):
        self.cluster = FakeCluster(**kw)

    def info(self):
        return {"version": {"number": "3.8.0"}}


def _client_with(fake):
    app.dependency_overrides[get_search_client] = lambda: fake
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_health_ok_when_cluster_green():
    r = _client_with(FakeClient()).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["opensearch"] == {"status": "green", "version": "3.8.0", "nodes": 1, "error": None}
    assert "Not medical advice" in body["disclaimer"]


def test_health_degraded_and_503_when_cluster_unreachable():
    r = _client_with(FakeClient(fail=True)).get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"
    assert r.json()["opensearch"]["status"] == "unavailable"


def test_health_degraded_when_cluster_red():
    r = _client_with(FakeClient(status="red")).get("/health")
    assert r.status_code == 503


def test_favicon_is_served_as_svg_when_built_and_linked_from_both_pages():
    from pathlib import Path

    built = Path("src/medical_rag/api/static/app/favicon.svg")
    r = TestClient(app).get("/favicon.svg")
    if built.exists():
        assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml")
        assert b"<svg" in r.content
    else:  # CI has no frontend build
        assert r.status_code == 404
    assert 'href="favicon.svg"' in Path("src/medical_rag/api/static/index.html").read_text(encoding="utf-8")
    assert 'href="/favicon.svg"' in Path("frontend/index.html").read_text(encoding="utf-8")
