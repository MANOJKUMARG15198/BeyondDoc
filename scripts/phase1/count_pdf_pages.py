"""Fill pdf_page_count in a batch's downloaded manifest (and LiverTox chapter PDFs inside the archive).

.venv/Scripts/python scripts/phase1/count_pdf_pages.py --batch batch_01
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import tarfile
from pathlib import Path

from pypdf import PdfReader

MANIFESTS = Path("data/manifests/batches")


def pages(data: bytes) -> int:
    return len(PdfReader(io.BytesIO(data)).pages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", default="batch_01")
    args = parser.parse_args()

    path = MANIFESTS / f"{args.batch}_downloaded.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    failures = 0
    for r in rows:
        if r["format"] == "pdf":
            try:
                r["pdf_page_count"] = pages(Path(r["file"]).read_bytes())
            except Exception as exc:  # recorded, not fatal
                r["pdf_page_count"] = f"error: {type(exc).__name__}"
                failures += 1
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    members_path = MANIFESTS / f"{args.batch}_livertox_members.csv"
    if members_path.exists():
        members = list(csv.DictReader(members_path.open(encoding="utf-8")))
        with tarfile.open(members[0]["archive"], "r:gz") as tf:
            for m in members:
                if m.get("member_pdf"):
                    m["pdf_page_count"] = pages(tf.extractfile(m["member_pdf"]).read())
        with members_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(members[0].keys()))
            writer.writeheader()
            writer.writerows(members)

    counted = [int(r["pdf_page_count"]) for r in rows if str(r.get("pdf_page_count", "")).isdigit()]
    print(f"PDFs counted: {len(counted)}, pages: {sum(counted)}, errors: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
