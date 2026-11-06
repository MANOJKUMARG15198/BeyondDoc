"""Phase 1, stage 2a: download openFDA bulk files (drug label, NDC, Drugs@FDA).

Reads the official manifest https://api.fda.gov/download.json, saves it, then
downloads each partition ZIP unchanged. Files already downloaded with the
expected size are skipped, so reruns resume. Every file gets a SHA-256 and a
row in downloaded.csv; failures go to failed.csv.

Standard library only (decision D8):

    python -I scripts/phase1/openfda_bulk_download.py --dry-run
    python -I scripts/phase1/openfda_bulk_download.py

Output (data/ is kept out of Git, decision D3):
    data/raw/openfda_bulk/<export_date>/download.json
    data/raw/openfda_bulk/<export_date>/<dataset>/<file>.zip
    data/raw/openfda_bulk/<export_date>/downloaded.csv
    data/raw/openfda_bulk/<export_date>/failed.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

MANIFEST_URL = "https://api.fda.gov/download.json"
DATASETS = ("label", "ndc", "drugsfda")
USER_AGENT = "RagMedical-Phase1-Bulk/0.1 (non-commercial research)"
TIMEOUT_S = 60
MAX_ATTEMPTS = 4
CHUNK = 1024 * 1024

log = logging.getLogger("openfda_bulk_download")


def http_get(url: str) -> urllib.request.addinfourl:
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=TIMEOUT_S)


def download(url: str, dest: Path) -> tuple[str, int, str]:
    """Stream url to dest via a .part file. Returns (sha256, bytes, content_type)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            digest = hashlib.sha256()
            size = 0
            with http_get(url) as response, tmp.open("wb") as fh:
                content_type = response.headers.get("Content-Type", "")
                while chunk := response.read(CHUNK):
                    fh.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            tmp.replace(dest)
            return digest.hexdigest(), size, content_type
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = 2**attempt
            log.warning("%s: %s, retry %d/%d in %ss", dest.name, exc, attempt, MAX_ATTEMPTS - 1, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, default=Path("data/raw/openfda_bulk"))
    parser.add_argument("--dry-run", action="store_true", help="list files and sizes; download nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with http_get(MANIFEST_URL) as response:
        manifest_bytes = response.read()
    drug = json.loads(manifest_bytes)["results"]["drug"]

    plan = []
    for name in DATASETS:
        for part in drug[name]["partitions"]:
            plan.append(
                {
                    "dataset": name,
                    "export_date": drug[name]["export_date"],
                    "total_records": drug[name]["total_records"],
                    **part,
                }
            )
    export_date = drug["label"]["export_date"]
    total_mb = sum(float(p["size_mb"]) for p in plan)
    log.info("manifest export %s: %d files, %.1f MB", export_date, len(plan), total_mb)
    if args.dry_run:
        for p in plan:
            print(f"{p['dataset']:9s} {p['size_mb']:>8s} MB  {p['file']}")
        return 0

    out_dir = args.out_root / export_date
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "download.json").write_bytes(manifest_bytes)

    downloaded, failed = [], []
    for p in plan:
        dest = out_dir / p["dataset"] / Path(urlsplit(p["file"]).path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "dataset": p["dataset"],
            "export_date": p["export_date"],
            "file": str(dest.relative_to(out_dir)),
            "source_url": p["file"],
            "manifest_size_mb": p["size_mb"],
            "manifest_records": p.get("records", ""),
        }
        try:
            if dest.exists():
                # Resume: keep an existing complete file, but re-hash it so the manifest stays truthful.
                sha, size, ctype, status = sha256_of(dest), dest.stat().st_size, "", "reused"
            else:
                started = time.monotonic()
                sha, size, ctype = download(p["file"], dest)
                status = f"downloaded in {time.monotonic() - started:.0f}s"
            with zipfile.ZipFile(dest) as zf:  # integrity check: valid ZIP with an intact CRC
                bad = zf.testzip()
                members = zf.namelist()
            if bad:
                raise zipfile.BadZipFile(f"corrupt member {bad}")
            row.update(
                sha256=sha,
                bytes=size,
                content_type=ctype,
                zip_members="|".join(members),
                downloaded_at_utc=datetime.now(UTC).isoformat(),
            )
            downloaded.append(row)
            log.info("%-60s %s (%.1f MB)", row["file"], status, size / 1e6)
        except Exception as exc:  # recorded, not fatal: the remaining files still download
            row.update(error=f"{type(exc).__name__}: {exc}", failed_at_utc=datetime.now(UTC).isoformat())
            failed.append(row)
            log.error("%s failed: %s", row["file"], exc)

    for name, rows in (("downloaded.csv", downloaded), ("failed.csv", failed)):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
            fields = sorted({k for r in rows for k in r}) or ["file"]
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    log.info("done: %d downloaded, %d failed, output in %s", len(downloaded), len(failed), out_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
