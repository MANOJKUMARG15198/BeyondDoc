"""Phase 1, stage 1: openFDA label inventory (metadata counts only).

Runs a small, fixed set of openFDA *count* queries to show how human Rx/OTC
labels are spread across routes, pharmacologic classes, substances and dosage
forms. No label documents are downloaded. The output is used to plan the
stratified DailyMed candidate sample (decision D5, option A).

Standard library only (decision D8). Run with the installed Python:

    python -I scripts/phase1/openfda_inventory.py --dry-run   # print queries, no network
    python -I scripts/phase1/openfda_inventory.py             # run the queries

Optional: set OPENFDA_API_KEY in the environment. The key is never written to
disk or logs.

Output (data/ is kept out of Git, decision D3):
    data/discovery/openfda/<run_id>/raw/<query_key>.json   original responses
    data/discovery/openfda/<run_id>/stratum_counts.csv     flattened counts
    data/discovery/openfda/<run_id>/run_log.json           queries, timings, statuses
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

BASE_URL = "https://api.fda.gov"
USER_AGENT = "RagMedical-Phase1-Inventory/0.1 (non-commercial research)"

# openFDA allows 240 requests/minute; one request per second keeps us far below it.
REQUEST_INTERVAL_S = 1.0
TIMEOUT_S = 30
MAX_ATTEMPTS = 4
RETRY_STATUSES = {429, 500, 502, 503, 504}

PRODUCT_TYPES = {
    "rx": "HUMAN PRESCRIPTION DRUG",
    "otc": "HUMAN OTC DRUG",
}

log = logging.getLogger("openfda_inventory")


def build_queries() -> list[dict]:
    """Return the fixed query plan. Each entry has a stable key used as its file name."""
    queries: list[dict] = []
    for short, product_type in PRODUCT_TYPES.items():
        label_filter = f'openfda.product_type:"{product_type}"'
        # Totals: all labels, original-packager labels, labels lacking openfda annotations.
        queries.append(
            {
                "key": f"label_{short}_total",
                "endpoint": "/drug/label.json",
                "params": {"search": label_filter, "limit": "1"},
            }
        )
        queries.append(
            {
                "key": f"label_{short}_original_packager",
                "endpoint": "/drug/label.json",
                "params": {"search": f"{label_filter} AND openfda.is_original_packager:true", "limit": "1"},
            }
        )
        for field in ("route", "pharm_class_epc", "substance_name"):
            queries.append(
                {
                    "key": f"label_{short}_count_{field}",
                    "endpoint": "/drug/label.json",
                    "params": {"search": label_filter, "count": f"openfda.{field}.exact", "limit": "1000"},
                }
            )
        # The label endpoint has no dosage-form field; the NDC directory does.
        queries.append(
            {
                "key": f"ndc_{short}_count_dosage_form",
                "endpoint": "/drug/ndc.json",
                "params": {"search": f'product_type:"{product_type}"', "count": "dosage_form.exact", "limit": "1000"},
            }
        )
    queries.append(
        {
            "key": "label_missing_openfda",
            "endpoint": "/drug/label.json",
            "params": {"search": "_missing_:openfda", "limit": "1"},
        }
    )
    return queries


def build_url(query: dict, api_key: str | None) -> str:
    params = dict(query["params"])
    if api_key:
        params["api_key"] = api_key
    return f"{BASE_URL}{query['endpoint']}?{urllib.parse.urlencode(params)}"


def redact(url: str) -> str:
    """Remove the API key from a URL before it is logged or saved."""
    parts = urllib.parse.urlsplit(url)
    params = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query) if k != "api_key"]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(params)))


def fetch(url: str) -> tuple[int, bytes]:
    """GET with timeout and bounded retries. Returns (status, body).

    openFDA answers 404 when a search matches nothing; that is a valid result,
    not a failure, so it is returned rather than raised.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 404, exc.read()
            if exc.code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS:
                raise
            wait = 2**attempt
            log.warning("HTTP %s, retry %d/%d in %ss", exc.code, attempt, MAX_ATTEMPTS - 1, wait)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = 2**attempt
            log.warning("%s, retry %d/%d in %ss", exc, attempt, MAX_ATTEMPTS - 1, wait)
        time.sleep(wait)
    raise RuntimeError("unreachable")


def flatten(query: dict, payload: dict) -> list[dict]:
    """Turn one response into stratum_counts.csv rows."""
    if "count" in query["params"]:
        return [
            {"query_key": query["key"], "value": item["term"], "count": item["count"]}
            for item in payload.get("results", [])
        ]
    total = payload.get("meta", {}).get("results", {}).get("total", 0)
    return [{"query_key": query["key"], "value": "__total__", "count": total}]


def run(out_dir: Path, api_key: str | None) -> int:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    entries: list[dict] = []
    failures = 0

    for query in build_queries():
        raw_path = raw_dir / f"{query['key']}.json"
        entry = {"key": query["key"], "url": redact(build_url(query, None))}
        if raw_path.exists():
            # Resume: a saved response is reused, so reruns are idempotent.
            entry["status"] = "reused"
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            started = time.monotonic()
            try:
                status, body = fetch(build_url(query, api_key))
            except Exception as exc:  # recorded, not fatal: the rest of the plan still runs
                failures += 1
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                log.error("%s failed: %s", query["key"], exc)
                entries.append(entry)
                time.sleep(REQUEST_INTERVAL_S)
                continue
            entry.update(
                http_status=status,
                elapsed_s=round(time.monotonic() - started, 3),
                fetched_at_utc=datetime.now(UTC).isoformat(),
            )
            payload = {"meta": {"results": {"total": 0}}, "results": []} if status == 404 else json.loads(body)
            entry["status"] = "no_match" if status == 404 else "ok"
            raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            time.sleep(REQUEST_INTERVAL_S)
        entry["openfda_last_updated"] = payload.get("meta", {}).get("last_updated")
        rows.extend(flatten(query, payload))
        entries.append(entry)
        log.info("%-40s %s", query["key"], entry["status"])

    with (out_dir / "stratum_counts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["query_key", "value", "count"])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "run_log.json").write_text(
        json.dumps({"api_key_used": bool(api_key), "queries": entries, "failures": failures}, indent=2),
        encoding="utf-8",
    )
    log.info("done: %d queries, %d failures, output in %s", len(entries), failures, out_dir)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, default=Path("data/discovery/openfda"))
    parser.add_argument(
        "--run-id",
        default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        help="reuse an earlier run id to resume it",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the query plan; no network")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dry_run:
        for query in build_queries():
            print(f"{query['key']:40s} {build_url(query, None)}")
        return 0
    return run(args.out_root / args.run_id, os.environ.get("OPENFDA_API_KEY") or None)


if __name__ == "__main__":
    sys.exit(main())
