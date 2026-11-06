"""Phase 1: download an approved batch plan.

Reads data/manifests/batches/<batch>_plan.csv and saves every original file
unchanged under data/raw/<source>/... Each file is validated by type, hashed
(SHA-256) and logged to <batch>_downloaded.csv; failures go to
<batch>_failed.csv. Existing complete files are re-hashed and reused, so reruns
are idempotent and resume where they stopped.

Standard library only (decision D8). PDF page counts need a PDF library and
are filled in during the foundation stage.

    python -I scripts/phase1/download_batch.py --batch batch_01
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

USER_AGENT = "RagMedical-Phase1-Download/0.1 (non-commercial research)"
TIMEOUT_S = 60
MAX_ATTEMPTS = 4
WORKERS = 4
MIN_INTERVAL_S = {
    "dailymed.nlm.nih.gov": 0.5,
    "www.accessdata.fda.gov": 0.5,
    "pmc-oa-opendata.s3.amazonaws.com": 0.2,
    "ftp.ncbi.nlm.nih.gov": 0.5,
}
RAW = Path("data/raw")
MANIFESTS = Path("data/manifests/batches")

log = logging.getLogger("download_batch")
_host_locks: dict[str, threading.Lock] = {}
_host_last: dict[str, float] = {}
_guard = threading.Lock()


def throttle(host: str) -> None:
    """Per-host minimum interval so parallel workers stay polite."""
    with _guard:
        lock = _host_locks.setdefault(host, threading.Lock())
    with lock:
        wait = _host_last.get(host, 0) + MIN_INTERVAL_S.get(host, 0.5) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _host_last[host] = time.monotonic()


def fetch(url: str, dest: Path) -> dict:
    """Stream url to dest through a .part file; returns response facts."""
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        throttle(urlsplit(url).hostname or "")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r, tmp.open("wb") as fh:
                headers = dict(r.headers)
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                final_url = r.geturl()
            tmp.replace(dest)
            return {
                "sha256": digest.hexdigest(),
                "bytes": size,
                "final_url": final_url,
                "content_type": headers.get("Content-Type", ""),
                "dailymed_last_updated": headers.get("X-DAILYMED-LABEL-LAST-UPDATED", ""),
                "content_disposition": headers.get("Content-Disposition", ""),
            }
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == MAX_ATTEMPTS:
                raise
            log.warning("%s HTTP %s, retry %d", url, exc.code, attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            log.warning("%s %s, retry %d", url, exc, attempt)
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, kind: str) -> str:
    """Return '' when the file is the expected type, else the reason it is not."""
    head = path.read_bytes()[:512] if path.stat().st_size else b""
    if not head:
        return "empty file"
    if kind == "zip":
        if not zipfile.is_zipfile(path):
            return "not a ZIP"
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                return f"corrupt ZIP member {bad}"
            if not any(n.lower().endswith(".xml") for n in zf.namelist()):
                return "ZIP has no XML"
    elif kind == "pdf" and not head.startswith(b"%PDF"):
        return "not a PDF"
    elif kind == "xml" and not head.lstrip().startswith(b"<"):
        return "not XML"
    elif kind == "tar.gz":
        try:
            with tarfile.open(path, "r:gz") as tf:
                tf.getmembers()
        except tarfile.TarError as exc:
            return f"bad tar.gz: {exc}"
    return ""


def kind_of(url: str) -> str:
    u = url.lower()
    if "downloadzipfile" in u or u.endswith(".zip"):
        return "zip"
    if "downloadpdffile" in u or u.endswith(".pdf"):
        return "pdf"
    if u.endswith(".tar.gz"):
        return "tar.gz"
    return "xml" if u.endswith(".xml") else "other"


def dailymed_latest_version(set_id: str) -> str:
    url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}/history.json"
    throttle("dailymed.nlm.nih.gov")
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=TIMEOUT_S
    ) as r:
        data = json.load(r)["data"]
    return str(max(int(d["spl_version"]) for d in data)) if data else ""


def targets(item: dict) -> list[tuple[str, Path]]:
    """(url, destination) pairs for one plan item, following the agreed data/raw layout."""
    src, doc = item["source_name"], item["source_document_id"]
    urls = item["download_url"].split()
    if src == "dailymed":
        base = RAW / "dailymed" / doc / f"v{item['_version']}"
        return [(u, base / f"{doc}.{kind_of(u)}") for u in urls]
    if src == "fda_drugsatfda":
        appl, doc_id = doc.split(":")
        return [(u, RAW / "fda_drugsatfda" / appl / f"{doc_id}_{Path(urlsplit(u).path).name}") for u in urls]
    if src == "pmc_oa":
        return [(u, RAW / "pmc_oa" / doc / Path(urlsplit(u).path).name) for u in urls]
    if src == "nih_livertox":  # one shared archive for all chapters
        return [(u, RAW / "nih_livertox" / Path(urlsplit(u).path).name) for u in urls]
    raise ValueError(f"unknown source {src}")


def process(item: dict, shared: dict) -> tuple[list[dict], list[dict]]:
    ok, failed = [], []
    if item["source_name"] == "dailymed":
        try:
            item["_version"] = dailymed_latest_version(item["source_document_id"]) or item["version"]
        except Exception as exc:
            item["_version"] = item["version"]
            log.warning("history lookup failed for %s: %s", item["source_document_id"], exc)
    for url, dest in targets(item):
        row = {
            "plan_item": item["plan_item"],
            "source_name": item["source_name"],
            "source_document_id": item["source_document_id"],
            "logical_document_id": f"{item['source_name']}:{item['source_document_id']}",
            "file": dest.as_posix(),
            "format": kind_of(url),
            "source_url": url,
            "openfda_version": item.get("version", ""),
            "dailymed_version": item.get("_version", ""),
        }
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with shared["lock"]:
                in_flight = shared["files"].setdefault(dest, threading.Event())
                owner = not in_flight.is_set() and dest not in shared["owners"]
                shared["owners"].add(dest)
            if dest.exists():
                facts = {"sha256": sha256_of(dest), "bytes": dest.stat().st_size, "reused": True}
            elif owner:
                facts = fetch(url, dest)
                in_flight.set()
            else:  # another worker is downloading the shared file
                in_flight.wait()
                facts = {"sha256": sha256_of(dest), "bytes": dest.stat().st_size, "reused": True}
            problem = validate(dest, kind_of(url))
            if problem:
                raise ValueError(problem)
            row.update(facts, downloaded_at_utc=datetime.now(UTC).isoformat(), pdf_page_count="")
            ok.append(row)
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}", failed_at_utc=datetime.now(UTC).isoformat())
            failed.append(row)
            log.error("%s %s failed: %s", item["plan_item"], url, exc)
            with shared["lock"]:
                shared["files"].setdefault(dest, threading.Event()).set()
    return ok, failed


def livertox_members(archive: Path, items: list[dict]) -> list[dict]:
    """Record which archive member holds each planned chapter (no extraction: originals stay as is)."""
    rows = []
    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()
    for item in items:
        # Archive members are named by chapter title, e.g. livertox_NBK547852/Atorvastatin.nxml
        chapter = item["title"].removeprefix("LiverTox: ").strip()
        stem = re.sub(r"[^A-Za-z0-9]", "", chapter)
        hits = {
            ext: n
            for n in names
            for ext in ("nxml", "pdf")
            if re.fullmatch(rf"[^/]+/{re.escape(stem)}\.{ext}", n, re.IGNORECASE)
        }
        rows.append(
            {
                "plan_item": item["plan_item"],
                "source_document_id": item["source_document_id"],
                "archive": archive.as_posix(),
                "member_nxml": hits.get("nxml", ""),
                "member_pdf": hits.get("pdf", ""),
                "status": "found" if "nxml" in hits else "member not found",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ["plan_item"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", default="batch_01")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    items = list(csv.DictReader((MANIFESTS / f"{args.batch}_plan.csv").open(encoding="utf-8")))
    shared = {"lock": threading.Lock(), "files": {}, "owners": set()}
    downloaded, failed = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for ok, bad in pool.map(lambda it: process(it, shared), items):
            downloaded += ok
            failed += bad
    write_csv(MANIFESTS / f"{args.batch}_downloaded.csv", downloaded)
    write_csv(MANIFESTS / f"{args.batch}_failed.csv", failed)

    ltx_items = [i for i in items if i["source_name"] == "nih_livertox"]
    archives = {Path(r["file"]) for r in downloaded if r["source_name"] == "nih_livertox"}
    if ltx_items and archives:
        write_csv(MANIFESTS / f"{args.batch}_livertox_members.csv", livertox_members(archives.pop(), ltx_items))

    docs_ok = {r["logical_document_id"] for r in downloaded} - {r["logical_document_id"] for r in failed}
    log.info(
        "done: %d files ok, %d failed, %d/%d logical documents complete",
        len(downloaded),
        len(failed),
        len(docs_ok),
        len(items),
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
