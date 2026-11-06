"""Run manifest: what an evaluation run actually used, so a result can be reproduced or compared."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PACKAGES = (
    "ragas",
    "sentence-transformers",
    "transformers",
    "torch",
    "anthropic",
    "openai",
    "opensearch-py",
    "langchain-community",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "changed_files": len(status.splitlines()) if status else 0,
    }


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def index_state(client: Any, alias: str) -> dict[str, Any]:
    """Concrete index behind the alias and its document count; never raises."""
    try:
        concrete = sorted(client.indices.get_alias(name=alias))
        return {"alias": alias, "indices": concrete, "docs": client.count(index=alias)["count"]}
    except Exception as exc:  # noqa: BLE001 - the manifest records the failure instead of stopping the run
        return {"alias": alias, "error": type(exc).__name__}


def run_manifest(*, testset: Path, client: Any, alias: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "testset": {"path": str(testset), "sha256": sha256_file(testset)},
        "index": index_state(client, alias),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "config": config,
    }
