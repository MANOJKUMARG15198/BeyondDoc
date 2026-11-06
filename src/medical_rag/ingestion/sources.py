"""Raw-document loading from a batch manifest: checksum-verified, read in memory, never written.

One RawDocument per logical document: the DailyMed SPL XML from its ZIP, the PMC JATS XML, the FDA letter PDF,
or a LiverTox NXML member from the shared tar archive. Other representations of the same document (PDFs) are
listed as alternate files and never parsed, so no document is chunked twice.
"""

from __future__ import annotations

import csv
import hashlib
import io
import tarfile
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

MANIFESTS = Path("data/manifests/batches")
PRIMARY_FORMAT = {"dailymed": "zip", "pmc_oa": "xml", "fda_drugsatfda": "pdf"}


class ChecksumError(ValueError):
    """A raw file does not match the SHA-256 recorded in its manifest."""


@dataclass(frozen=True, slots=True)
class RawDocument:
    logical_document_id: str
    source: str
    source_document_id: str
    raw_file: str
    raw_member: str | None
    raw_sha256: str  # of raw_file (for LiverTox: the archive)
    source_url: str
    alternate_files: tuple[str, ...]
    data: bytes
    catalog: dict[str, str] = field(default_factory=dict)  # the batch plan row (curated title, drug), if any


def read_verified(path: str | Path, sha256: str) -> bytes:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ChecksumError(f"checksum mismatch: {path}")
    return data


def iter_documents(
    select: Callable[[str, str], bool] = lambda source, source_document_id: True,
    *,
    batch: str = "batch_01",
    manifests: Path = MANIFESTS,
) -> Iterator[RawDocument]:
    """Yield the selected documents of a batch in manifest order (LiverTox chapters last)."""
    rows = list(csv.DictReader((manifests / f"{batch}_downloaded.csv").open(encoding="utf-8")))
    plan_file = manifests / f"{batch}_plan.csv"
    plan = {
        r["source_document_id"]: {"title": r.get("title", ""), "drug": r.get("drug", "")}
        for r in (csv.DictReader(plan_file.open(encoding="utf-8")) if plan_file.exists() else [])
    }
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["logical_document_id"]].append(row)
    for logical, group in groups.items():
        source, sid = group[0]["source_name"], group[0]["source_document_id"]
        if source not in PRIMARY_FORMAT or not select(source, sid):
            continue
        primary = next(r for r in group if r["format"] == PRIMARY_FORMAT[source])
        data, member = read_verified(primary["file"], primary["sha256"]), None
        if source == "dailymed":
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                member = next(n for n in zf.namelist() if n.lower().endswith(".xml"))
                data = zf.read(member)
        alternates = tuple(r["file"] for r in group if r is not primary)
        yield RawDocument(
            logical,
            source,
            sid,
            primary["file"],
            member,
            primary["sha256"],
            primary["source_url"],
            alternates,
            data,
            plan.get(sid, {}),
        )

    archives = [r for r in rows if r["source_name"] == "nih_livertox"]
    members_file = manifests / f"{batch}_livertox_members.csv"
    if not archives or not members_file.exists():
        return
    members = [
        m
        for m in csv.DictReader(members_file.open(encoding="utf-8"))
        if m.get("status", "found") == "found" and select("nih_livertox", m["source_document_id"])
    ]
    if not members:
        return
    archive = archives[0]
    blob = read_verified(archive["file"], archive["sha256"])
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in members:
            extracted = tf.extractfile(m["member_nxml"])
            if extracted is None:
                raise ValueError(f"LiverTox member not found: {m['member_nxml']}")
            yield RawDocument(
                f"nih_livertox:{m['source_document_id']}",
                "nih_livertox",
                m["source_document_id"],
                archive["file"],
                m["member_nxml"],
                archive["sha256"],
                archive["source_url"],
                tuple(p for p in (m.get("member_pdf"),) if p),
                extracted.read(),
                plan.get(m["source_document_id"], {}),
            )
