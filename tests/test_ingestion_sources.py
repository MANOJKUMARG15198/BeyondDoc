"""Raw loading from a synthetic manifest and synthetic raw files under tmp_path."""

import csv
import hashlib
import io
import tarfile
import zipfile

import pytest

from medical_rag.ingestion.sources import ChecksumError, iter_documents, read_verified

FIELDS = ["logical_document_id", "source_name", "source_document_id", "file", "format", "source_url", "sha256"]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def batch(tmp_path):
    raw = tmp_path / "raw"
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("label.xml", "<document/>")
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode="w:gz") as tf:
        info = tarfile.TarInfo("lt/Drug.nxml")
        payload = b"<book-part-wrapper/>"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    rows = [
        ("dailymed:s1", "dailymed", "s1", raw / "s1.zip", "zip", zbuf.getvalue()),
        ("dailymed:s1", "dailymed", "s1", raw / "s1.pdf", "pdf", b"%PDF"),
        ("pmc_oa:PMC1.1", "pmc_oa", "PMC1.1", raw / "PMC1.xml", "xml", b"<article/>"),
        ("fda_drugsatfda:NDA1:2", "fda_drugsatfda", "NDA1:2", raw / "l.pdf", "pdf", b"%PDF-letter"),
        ("nih_livertox:archive", "nih_livertox", "archive", raw / "lt.tar.gz", "tar.gz", tbuf.getvalue()),
    ]
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    with (manifests / "batch_x_downloaded.csv").open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, FIELDS)
        w.writeheader()
        for logical, source, sid, path, fmt, data in rows:
            w.writerow(
                dict(
                    zip(
                        FIELDS,
                        [logical, source, sid, str(path), fmt, f"https://x.test/{sid}", write(path, data)],
                        strict=True,
                    )
                )
            )
    with (manifests / "batch_x_livertox_members.csv").open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, ["source_document_id", "member_nxml", "member_pdf", "status"])
        w.writeheader()
        w.writerow(
            {
                "source_document_id": "NBK1",
                "member_nxml": "lt/Drug.nxml",
                "member_pdf": "lt/Drug.pdf",
                "status": "found",
            }
        )
    return manifests, raw


def test_one_document_per_logical_id_with_alternates(batch):
    manifests, raw = batch
    docs = list(iter_documents(batch="batch_x", manifests=manifests))
    assert [d.logical_document_id for d in docs] == [
        "dailymed:s1",
        "pmc_oa:PMC1.1",
        "fda_drugsatfda:NDA1:2",
        "nih_livertox:NBK1",
    ]
    label, _, _, chapter = docs
    assert (label.raw_member, label.data, label.alternate_files) == (
        "label.xml",
        b"<document/>",
        (str(raw / "s1.pdf"),),
    )
    assert (chapter.raw_member, chapter.data, chapter.alternate_files) == (
        "lt/Drug.nxml",
        b"<book-part-wrapper/>",
        ("lt/Drug.pdf",),
    )


def test_selection(batch):
    manifests, _ = batch
    docs = list(iter_documents(lambda source, sid: source == "pmc_oa", batch="batch_x", manifests=manifests))
    assert [d.source_document_id for d in docs] == ["PMC1.1"]


def test_checksum_mismatch_raises_and_files_are_not_modified(batch):
    manifests, raw = batch
    before = {p: p.read_bytes() for p in raw.iterdir()}
    list(iter_documents(batch="batch_x", manifests=manifests))
    assert {p: p.read_bytes() for p in raw.iterdir()} == before
    (raw / "PMC1.xml").write_bytes(b"<article>changed</article>")
    with pytest.raises(ChecksumError):
        list(iter_documents(batch="batch_x", manifests=manifests))
    with pytest.raises(ChecksumError):
        read_verified(raw / "PMC1.xml", "0" * 64)
