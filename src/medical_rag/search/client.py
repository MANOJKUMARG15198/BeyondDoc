"""OpenSearch client factory and cluster health check."""

from __future__ import annotations

from opensearchpy import OpenSearch
from opensearchpy.exceptions import OpenSearchException

from medical_rag.core.config import Settings


def make_client(settings: Settings) -> OpenSearch:
    return OpenSearch(
        hosts=[str(settings.opensearch_url)],
        timeout=settings.opensearch_timeout_s,
        max_retries=1,
        retry_on_timeout=False,
    )


def cluster_health(client: OpenSearch) -> dict:
    """Return a small health summary; never raises for an unreachable cluster."""
    try:
        health = client.cluster.health()
        info = client.info()
    except (OpenSearchException, ConnectionError, OSError) as exc:
        return {"status": "unavailable", "error": type(exc).__name__}
    return {
        "status": health.get("status", "unknown"),
        "version": info.get("version", {}).get("number"),
        "nodes": health.get("number_of_nodes"),
    }
