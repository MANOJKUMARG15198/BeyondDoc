"""Integration checks against the local Docker OpenSearch. Run: pytest -m integration"""

import uuid

import pytest
from fastapi.testclient import TestClient

from medical_rag.api.main import app
from medical_rag.core.config import Settings
from medical_rag.search.client import cluster_health, make_client

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    c = make_client(Settings(_env_file=None))
    if cluster_health(c)["status"] not in ("green", "yellow"):
        pytest.skip("OpenSearch not running (docker compose up -d)")
    return c


def test_cluster_is_pinned_version(client):
    assert client.info()["version"]["number"] == "3.8.0"


def test_index_write_read_delete_roundtrip(client):
    index = f"medrag-it-{uuid.uuid4().hex[:8]}"
    try:
        client.indices.create(index=index, body={"settings": {"number_of_replicas": 0}})
        client.index(index=index, id="doc-1", body={"text": "synthetic fixture"}, refresh=True)
        assert client.get(index=index, id="doc-1")["_source"]["text"] == "synthetic fixture"
    finally:
        client.indices.delete(index=index, ignore_unavailable=True)


def test_health_endpoint_against_real_cluster(client):
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["opensearch"]["version"] == "3.8.0"
