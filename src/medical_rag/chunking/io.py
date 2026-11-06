"""JSONL and manifest output for chunking experiments (P2-11 layout: data/processed/chunking/experiments/<name>/)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Write one JSON object per line (UTF-8, keys in model order); returns the number of records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for record in records:
            out.write(record.model_dump_json() + "\n")
            n += 1
    return n


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
