"""Phase 1, Batch 1 plan: 100 documents linked around one set of drugs.

Builds data/manifests/batches/batch_01_plan.csv for user approval. Downloads
nothing except small metadata responses (E-utilities, PMC bucket listings).

    70 DailyMed labels   50 Rx (30 popular + 20 with boxed warning, distinct classes), 20 OTC popular
    10 Drugs@FDA docs    original-approval Summary Review, else approval Letter, for brand Rx drugs above
    15 PMC OA articles   one per Rx drug above; CC BY / CC0 only (other licenses wait for decision P1)
     5 LiverTox chapters one per Rx drug above; original = NCBI litarch book archive

    python -I scripts/phase1/plan_batch_01.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

CANDIDATES = Path("data/discovery/dailymed_stage2/candidates_dailymed.csv")
DRUGSFDA_ZIP = Path("data/raw/openfda_bulk/2026-11-02/drugsfda/drug-drugsfda-0001-of-0001.json.zip")
OUT = Path("data/manifests/batches/batch_01_plan.csv")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# NCBI asks E-utilities clients to send a contact email; set MEDRAG_NCBI_EMAIL to your own (never committed).
EUTILS_COMMON = {"tool": "RagMedical", "retmode": "json"}
if os.environ.get("MEDRAG_NCBI_EMAIL"):
    EUTILS_COMMON["email"] = os.environ["MEDRAG_NCBI_EMAIL"]
PMC_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
LIVERTOX_ARCHIVE = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/29/31/livertox_NBK547852.tar.gz"
ACCEPTED_PMC_LICENSES = {"CC BY", "CC0"}
# Notices are not articles; case reports are left out pending a user decision on patient cases.
EXCLUDED_TITLE = re.compile(
    r"^(comment|reply|response|correction|publisher correction|erratum|corrigendum|"
    r"retraction|expression of concern)\b|case report|case series",
    re.IGNORECASE,
)
USER_AGENT = "RagMedical-Phase1-Batch1/0.1 (non-commercial research)"
EUTILS_INTERVAL_S = 0.4  # 3 requests/second without an API key

log = logging.getLogger("plan_batch_01")


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as exc:
            if attempt == 3:
                raise
            log.warning("%s: %s, retrying", url[:80], exc)
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def eutils(endpoint: str, **params) -> dict:
    time.sleep(EUTILS_INTERVAL_S)
    return json.loads(get(f"{EUTILS}/{endpoint}.fcgi?{urllib.parse.urlencode({**EUTILS_COMMON, **params})}"))


def drug_word(c: dict) -> str:
    """Search term for a single-ingredient label: first word of its generic name."""
    return re.split(r"[\s,]+", c["generic_name"].split("|")[0].strip().lower())[0]


def single(c: dict) -> bool:
    return "|" not in c["active_ingredient"] and bool(c["generic_name"])


def select_dailymed(cands: list[dict]) -> list[dict]:
    eligible = [c for c in cands if not c["cap_category"]]

    def by_size(c: dict) -> int:
        return -int(c["group_member_count"])

    rx = sorted((c for c in eligible if c["document_type"] == "HUMAN PRESCRIPTION DRUG"), key=by_size)
    otc = sorted((c for c in eligible if c["document_type"] == "HUMAN OTC DRUG"), key=by_size)
    picked, seen_words, seen_classes = [], set(), set()

    def add(c: dict, reason: str) -> None:
        picked.append({**c, "batch_reason": reason})
        seen_words.add(drug_word(c) if c["generic_name"] else c["active_ingredient"])

    for c in rx:  # 30 popular Rx, distinct drugs
        w = drug_word(c) if c["generic_name"] else c["active_ingredient"]
        if sum(p["batch_reason"] == "rx_popular" for p in picked) < 30 and w not in seen_words:
            add(c, "rx_popular")
            seen_classes.add(c["medicine_category"].split("|")[0])
    for c in rx:  # 20 Rx with boxed warning, distinct classes
        klass = c["medicine_category"].split("|")[0]
        w = drug_word(c) if c["generic_name"] else c["active_ingredient"]
        if (
            sum(p["batch_reason"] == "rx_boxed_warning" for p in picked) < 20
            and c["has_boxed_warning"] == "True"
            and w not in seen_words
            and (not klass or klass not in seen_classes)
        ):
            add(c, "rx_boxed_warning")
            if klass:
                seen_classes.add(klass)
    for c in otc:  # 20 popular OTC, distinct drugs
        w = drug_word(c) if c["generic_name"] else c["active_ingredient"]
        if sum(p["batch_reason"] == "otc_popular" for p in picked) < 20 and w not in seen_words:
            add(c, "otc_popular")
    return picked


def plan_fda(dm: list[dict], n: int = 10) -> list[dict]:
    with zipfile.ZipFile(DRUGSFDA_ZIP) as zf:
        apps = {a["application_number"]: a for a in json.load(zf.open(zf.namelist()[0]))["results"]}
    rows = []
    for c in dm:
        if len(rows) >= n or c["is_brand"] != "True" or c["document_type"] != "HUMAN PRESCRIPTION DRUG":
            continue
        for appl in c["application_number"].split("|"):
            app = apps.get(appl)
            if not app:
                continue
            # Some Drugs@FDA URLs are malformed (e.g. contain spaces); those are skipped.
            pdfs = [
                (s, d)
                for s in app.get("submissions", [])
                for d in s.get("application_docs") or []
                if (d.get("url") or "").lower().endswith(".pdf") and not re.search(r"\s", d["url"])
            ]
            orig = [(s, d) for s, d in pdfs if s.get("submission_type") == "ORIG"]
            # Preference: original-approval Summary Review, original Letter, then the latest Letter.
            latest = sorted(pdfs, key=lambda sd: sd[1].get("date", ""), reverse=True)
            for wanted, pool in (("Summary Review", orig), ("Letter", orig), ("Letter", latest)):
                hit = next(((s, d) for s, d in pool if d.get("type") == wanted), None)
                if hit:
                    s, d = hit
                    rows.append(
                        {
                            "source_name": "fda_drugsatfda",
                            "source_document_id": f"{appl}:{d['id']}",
                            "title": f"{wanted} — {c['brand_name'].split('|')[0]} ({appl}, "
                            f"{s.get('submission_type', '')} {s.get('submission_number', '')})",
                            "drug": drug_word(c),
                            "source_url": d["url"],
                            "download_url": d["url"],
                            "document_type": wanted,
                            "license": "",
                            "license_basis": "source_policy",
                            "formats": "PDF",
                            "publication_date": d.get("date", ""),
                            "batch_reason": f"linked to dailymed:{c['source_document_id']}",
                        }
                    )
                    break
            if rows and rows[-1]["drug"] == drug_word(c):
                break
    return rows


def pmc_versions(pmcid: str) -> list[str]:
    xml = get(f"{PMC_BUCKET}/?list-type=2&prefix=metadata/{pmcid}.&max-keys=50")
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [k.text for k in ElementTree.fromstring(xml).findall(".//s3:Key", ns)]
    return sorted(keys, key=lambda k: int(k.rsplit(".", 2)[1]))


def plan_pmc(drugs: list[str], n: int = 15) -> list[dict]:
    rows, seen = [], set()
    for drug in drugs:
        if len(rows) >= n:
            break
        # PMC has no publication-type field ([pt] maps to All Fields), so title terms pick
        # reviews, trials and safety/PK studies, and veterinary titles are excluded.
        term = (
            f'{drug}[ti] AND (review[ti] OR "meta-analysis"[ti] OR trial[ti] OR safety[ti] '
            f"OR pharmacokinetics[ti] OR efficacy[ti]) NOT (dog[ti] OR dogs[ti] OR canine[ti] OR cat[ti] "
            f'OR cats[ti] OR feline[ti] OR equine[ti] OR veterinary[ti]) AND "cc by license"[filter]'
        )
        ids = eutils("esearch", db="pmc", retmax="20", term=term)["esearchresult"]["idlist"]
        for uid in ids:
            pmcid = f"PMC{uid}"
            keys = pmc_versions(pmcid)
            if not keys or pmcid in seen:
                continue
            meta = json.loads(get(f"{PMC_BUCKET}/{keys[-1]}"))
            if meta.get("is_retracted") or meta.get("license_code") not in ACCEPTED_PMC_LICENSES:
                continue
            title = meta.get("title") or ""
            if EXCLUDED_TITLE.search(title) or drug not in title.lower():
                continue
            seen.add(pmcid)
            files = [
                u.split("?")[0].replace("s3://pmc-oa-opendata", PMC_BUCKET)
                for u in (meta.get("xml_url"), meta.get("pdf_url"))
                if u
            ]
            rows.append(
                {
                    "source_name": "pmc_oa",
                    "source_document_id": f"{pmcid}.{meta['version']}",
                    "title": meta.get("title", ""),
                    "drug": drug,
                    "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                    "download_url": " ".join(files),
                    "document_type": "research_article",
                    "license": meta["license_code"],
                    "license_basis": "document",
                    "formats": "|".join(f.rsplit(".", 1)[1].upper() for f in files),
                    "publication_date": "",
                    "doi": meta.get("doi") or "",
                    "pmid": meta.get("pmid") or "",
                    "batch_reason": f"pmc search {drug}[ti], first eligible result",
                }
            )
            break
    return rows


def plan_livertox(drugs: list[str], n: int = 5) -> list[dict]:
    rows = []
    for drug in drugs:
        if len(rows) >= n:
            break
        ids = eutils("esearch", db="books", term=f"livertox[book] AND {drug}[title]")["esearchresult"]["idlist"]
        for uid in ids:
            s = eutils("esummary", db="books", id=uid)["result"][uid]
            if s.get("rtype") == "chapter" and s.get("title", "").lower().startswith(drug):
                acc = s["chapteraccessionid"]
                rows.append(
                    {
                        "source_name": "nih_livertox",
                        "source_document_id": acc,
                        "title": f"LiverTox: {s['title']}",
                        "drug": drug,
                        "source_url": f"https://www.ncbi.nlm.nih.gov/books/{acc}/",
                        "download_url": LIVERTOX_ARCHIVE,
                        "document_type": "reference_chapter",
                        "license": "public domain (per Bookshelf notice; to verify)",
                        "license_basis": "source_policy",
                        "formats": "NXML (in book archive)",
                        "publication_date": "",
                        "batch_reason": "LiverTox chapter for a batch drug",
                    }
                )
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite an existing plan (search results can drift)")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT} exists; it may already be approved and downloaded. Use --force to rebuild.")
        return 1
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cands = list(csv.DictReader(CANDIDATES.open(encoding="utf-8")))
    dm = select_dailymed(cands)
    log.info("DailyMed selected: %d", len(dm))
    rx_single = [c for c in dm if c["document_type"] == "HUMAN PRESCRIPTION DRUG" and single(c)]
    drugs = list(dict.fromkeys(drug_word(c) for c in rx_single))

    rows = [
        {
            "source_name": "dailymed",
            "source_document_id": c["source_document_id"],
            "title": c["brand_name"].split("|")[0] + " — " + c["generic_name"].split("|")[0],
            "drug": drug_word(c) if c["generic_name"] else "",
            "source_url": c["source_url"],
            "download_url": c["download_url"] + " " + c["download_url"].replace("downloadzipfile", "downloadpdffile"),
            "document_type": c["document_type"],
            "license": "",
            "license_basis": "source_policy",
            "formats": "SPL XML (ZIP)|PDF",
            "publication_date": "",
            "effective_time": c["effective_time"],
            "version": c["version"],
            "duplicate_group_id": c["duplicate_group_id"],
            "has_boxed_warning": c["has_boxed_warning"],
            "batch_reason": c["batch_reason"],
        }
        for c in dm
    ]
    fda = plan_fda(dm)
    log.info("FDA docs: %d", len(fda))
    pmc = plan_pmc(drugs)
    log.info("PMC articles: %d", len(pmc))
    ltx = plan_livertox(drugs)
    log.info("LiverTox chapters: %d", len(ltx))
    rows += fda + pmc + ltx

    for i, r in enumerate(rows, start=1):
        r["plan_item"] = f"b01-{i:03d}"
        r["collection_status"] = "planned"
    fields = [
        "plan_item",
        "source_name",
        "source_document_id",
        "title",
        "drug",
        "document_type",
        "batch_reason",
        "source_url",
        "download_url",
        "formats",
        "license",
        "license_basis",
        "publication_date",
        "effective_time",
        "version",
        "doi",
        "pmid",
        "duplicate_group_id",
        "has_boxed_warning",
        "collection_status",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("plan: %d items -> %s", len(rows), OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
