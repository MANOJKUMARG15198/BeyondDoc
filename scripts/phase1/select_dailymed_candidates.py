"""Phase 1, stage 2b: choose DailyMed label candidates from the openFDA bulk export.

Local files only, no network. Applies the confirmed rules in
docs/data/SOURCE_REGISTER.md:
    D6   representative = brand (NDA/BLA) > brand owner's own label > newest > most complete
    D6b  no brand label in a group -> newest generic, then most complete
    D6c  group = Rx/OTC + substances + routes + dosage forms
    D6d  flag allergen/excipient differences between group members
    D9   caps (homeopathic 10, sunscreen 5, hand sanitizer 5, medical gas 1, allergen 5),
         ~500 Rx / ~200 OTC; candidates are over-sampled 1.3x

Values are copied from source fields only. Anything a rule cannot decide from
those fields goes to review_flags.csv instead of being guessed.

    python -I scripts/phase1/select_dailymed_candidates.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

RX, OTC = "HUMAN PRESCRIPTION DRUG", "HUMAN OTC DRUG"
TARGETS = {RX: 650, OTC: 260}  # 1.3 x (500 Rx, 200 OTC)
CAPS = {"homeopathic": 10, "sunscreen": 5, "hand_sanitizer": 5, "medical_gas": 1, "allergen": 5}

SUNSCREEN_SUBSTANCES = {
    "AVOBENZONE",
    "OCTISALATE",
    "OCTOCRYLENE",
    "HOMOSALATE",
    "OXYBENZONE",
    "OCTINOXATE",
    "ZINC OXIDE",
    "TITANIUM DIOXIDE",
    "ENSULIZOLE",
}
SANITIZER_SUBSTANCES = {"ALCOHOL", "ISOPROPYL ALCOHOL", "BENZALKONIUM CHLORIDE"}
# Many medical gases carry an NDA, so the marketing category alone does not catch them.
MEDICAL_GAS_SUBSTANCES = {"OXYGEN", "NITROGEN", "NITROUS OXIDE", "CARBON DIOXIDE", "HELIUM", "AIR"}

# Sections used for the completeness tie-breaker; alternatives count once.
COMPLETENESS = {
    RX: [
        ("boxed_warning",),
        ("indications_and_usage",),
        ("dosage_and_administration",),
        ("dosage_forms_and_strengths",),
        ("contraindications",),
        ("warnings_and_cautions", "warnings", "precautions"),
        ("adverse_reactions",),
        ("drug_interactions",),
        ("use_in_specific_populations", "pregnancy"),
        ("clinical_pharmacology", "mechanism_of_action", "pharmacokinetics"),
        ("overdosage",),
        ("how_supplied",),
    ],
    OTC: [
        ("purpose",),
        ("active_ingredient",),
        ("indications_and_usage",),
        ("warnings",),
        ("do_not_use",),
        ("ask_doctor", "ask_doctor_or_pharmacist"),
        ("stop_use",),
        ("dosage_and_administration",),
        ("inactive_ingredient",),
    ],
}

# D6d excipient terms, matched case-insensitively in ingredient-bearing sections.
EXCIPIENTS = {
    "lactose": r"\blactose\b",
    "gluten_wheat": r"\bgluten\b|\bwheat\b",
    "tartrazine": r"tartrazine|yellow\s*(?:no\.?|#)?\s*5\b",
    "sulfites": r"sulfite|metabisulfite|bisulfite",
    "peanut": r"peanut|arachis",
    "soy": r"\bsoy(?:a|bean)?\b",
    "gelatin": r"gelatin",
    "aspartame": r"aspartame|phenylalanine",
    "benzyl_alcohol": r"benzyl alcohol",
    "sesame": r"sesame",
}
EXCIPIENT_RES = {k: re.compile(v, re.IGNORECASE) for k, v in EXCIPIENTS.items()}
INGREDIENT_SECTIONS = ("inactive_ingredient", "description", "spl_product_data_elements")

log = logging.getLogger("select_dailymed_candidates")


def joined(values) -> str:
    return "|".join(sorted({v.strip() for v in values or [] if v and v.strip()}))


def load_ndc(path: Path) -> tuple[dict, dict]:
    """Index NDC products by product_ndc and by spl_id (the label version id)."""
    with zipfile.ZipFile(path) as zf:
        data = json.load(zf.open(zf.namelist()[0]))
    by_ndc, by_spl = {}, defaultdict(list)
    for p in data["results"]:
        info = (p.get("dosage_form") or "", p.get("marketing_category") or "")
        if p.get("product_ndc"):
            by_ndc[p["product_ndc"]] = info
        if p.get("spl_id"):
            by_spl[p["spl_id"]].append(info)
    log.info("NDC products indexed: %d", len(by_ndc))
    return by_ndc, by_spl


def extract(label: dict, by_ndc: dict, by_spl: dict, source_file: str) -> dict:
    o = label.get("openfda") or {}
    product_types = sorted(set(o.get("product_type") or []))
    infos = [by_ndc[n] for n in o.get("product_ndc") or [] if n in by_ndc] or by_spl.get(label.get("id"), [])
    ptype = product_types[0] if len(product_types) == 1 else ""
    present = {k for k, v in label.items() if v}
    score = sum(any(s in present for s in alts) for alts in COMPLETENESS.get(ptype, []))
    text = " ".join(" ".join(label.get(s) or []) for s in INGREDIENT_SECTIONS)
    return {
        "set_id": label.get("set_id", ""),
        "spl_id": label.get("id", ""),
        "version": label.get("version", ""),
        "effective_time": label.get("effective_time", ""),
        "has_openfda": bool(o),
        "product_type": joined(product_types),
        "substances": joined(o.get("substance_name")),
        "routes": joined(o.get("route")),
        "dosage_forms": joined(i[0] for i in infos),
        "marketing_categories": joined(i[1] for i in infos),
        "application_numbers": joined(o.get("application_number")),
        "is_original_packager": joined(str(v) for v in o.get("is_original_packager") or []),
        "brand_name": joined(o.get("brand_name")),
        "generic_name": joined(o.get("generic_name")),
        "manufacturer_name": joined(o.get("manufacturer_name")),
        "pharm_class_epc": joined(o.get("pharm_class_epc")),
        "unii": joined(o.get("unii")),
        "product_ndcs": joined(o.get("product_ndc")),
        "completeness_score": score,
        "has_boxed_warning": "boxed_warning" in present,
        "excipients": joined(k for k, rx in EXCIPIENT_RES.items() if rx.search(text)),
        "source_file": source_file,
    }


def is_brand(r: dict) -> bool:
    # Brand = NDA/BLA marketing category. "NDA AUTHORIZED GENERIC" carries an NDA number but is a generic.
    cats = set(r["marketing_categories"].split("|")) - {""}
    if cats:
        return bool(cats & {"NDA", "BLA"})
    return any(a.startswith(("NDA", "BLA")) for a in r["application_numbers"].split("|") if a)


def cap_category(r: dict) -> str:
    subs = set(r["substances"].split("|")) - {""}
    topical = "TOPICAL" in r["routes"].split("|")
    cats = r["marketing_categories"]
    if "HOMEOPATHIC" in cats:
        return "homeopathic"
    if "MEDICAL GAS" in cats or (subs and subs <= MEDICAL_GAS_SUBSTANCES):
        return "medical_gas"
    if "allergen" in r["pharm_class_epc"].lower():
        return "allergen"
    if subs and topical and subs <= SUNSCREEN_SUBSTANCES:
        return "sunscreen"
    if subs and topical and subs <= SANITIZER_SUBSTANCES:
        return "hand_sanitizer"
    return ""


def pick_representative(members: list[dict]) -> tuple[dict, str]:
    """D6 / D6b. Returns (representative, reason)."""
    brands = [m for m in members if is_brand(m)]
    if brands:
        pool, reason = brands, "brand"
        originals = [m for m in pool if m["is_original_packager"] == "True"]
        if originals:
            pool, reason = originals, "brand, original packager"
    else:
        pool, reason = members, "no brand label: newest generic"
    rep = max(pool, key=lambda m: (m["effective_time"], m["completeness_score"], m["set_id"]))
    return rep, f"{reason}; newest effective_time; then completeness"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0].keys()) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bulk-dir", type=Path, default=Path("data/raw/openfda_bulk/2026-11-02"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/discovery/dailymed_stage2"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(UTC).isoformat()
    export_date = args.bulk_dir.name

    by_ndc, by_spl = load_ndc(next((args.bulk_dir / "ndc").glob("*.zip")))

    # 1-2. Extract metadata from every label.
    labels: list[dict] = []
    for path in sorted((args.bulk_dir / "label").glob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            data = json.load(zf.open(zf.namelist()[0]))
        labels.extend(extract(rec, by_ndc, by_spl, path.name) for rec in data["results"])
        del data
        log.info("%s -> %d labels so far", path.name, len(labels))
    write_csv(args.out_dir / "labels_extracted.csv", labels)
    write_csv(
        args.out_dir / "unannotated_pool.csv",
        [lab for lab in labels if not lab["has_openfda"]],
        ["set_id", "spl_id", "version", "effective_time", "completeness_score", "source_file"],
    )

    # 3. Group annotated Rx/OTC labels; incomplete keys go to review.
    flags: list[dict] = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for lab in labels:
        if not lab["has_openfda"] or lab["product_type"] not in (RX, OTC):
            continue
        missing = [f for f in ("substances", "routes", "dosage_forms") if not lab[f]]
        if missing:
            flags.append(
                {"flag": "incomplete_group_key", "set_id": lab["set_id"], "detail": "missing " + ",".join(missing)}
            )
            continue
        key = " || ".join((lab["product_type"], lab["substances"], lab["routes"], lab["dosage_forms"]))
        groups[key].append(lab)
    log.info("groups: %d", len(groups))

    # 4-6. Representative, cap category and excipient flags per group.
    group_rows, member_rows = [], []
    for gid, (key, members) in enumerate(sorted(groups.items()), start=1):
        group_id = f"dmg-{gid:06d}"
        rep, reason = pick_representative(members)
        rep_exc = set(rep["excipients"].split("|")) - {""}
        for m in members:
            m_exc = set(m["excipients"].split("|")) - {""}
            diff = m_exc ^ rep_exc
            if m is not rep and diff:
                flags.append(
                    {
                        "flag": "excipient_difference",
                        "set_id": m["set_id"],
                        "group_id": group_id,
                        "detail": f"vs representative {rep['set_id']}: {joined(diff)}",
                    }
                )
            member_rows.append(
                {
                    "duplicate_group_id": group_id,
                    "set_id": m["set_id"],
                    "is_representative": m is rep,
                    "is_brand": is_brand(m),
                    "is_original_packager": m["is_original_packager"],
                    "effective_time": m["effective_time"],
                    "completeness_score": m["completeness_score"],
                    "manufacturer_name": m["manufacturer_name"],
                    "excipients": m["excipients"],
                    "selection_reason": reason if m is rep else "",
                }
            )
        group_rows.append(
            {
                "group_id": group_id,
                "key": key,
                "rep": rep,
                "size": len(members),
                "cap": cap_category(rep),
                "reason": reason,
            }
        )

    # 7. Diversity: round-robin over (primary class, primary route) buckets. Buckets with
    #    more labels (more widely marketed drugs) go first; inside a bucket, larger groups
    #    first, preferring substance sets not yet chosen. openFDA has no class for many
    #    common drugs (e.g. atorvastatin, metformin), so an unclassified substance set is
    #    its own bucket instead of sharing one "(no class)" bucket.
    selected: list[dict] = []
    cap_used: Counter = Counter()
    for ptype, target in TARGETS.items():
        buckets: dict[tuple, list] = defaultdict(list)
        for g in group_rows:
            if g["rep"]["product_type"] == ptype:
                r = g["rep"]
                klass = r["pharm_class_epc"].split("|")[0] or f"(no class) {r['substances']}"
                buckets[(klass, r["routes"].split("|")[0])].append(g)
        for b in buckets.values():
            b.sort(key=lambda g: (-g["size"], g["group_id"]))
        chosen_subs: set[str] = set()
        count = 0

        def take(g: dict, phase: str, chosen_subs: set[str] = chosen_subs) -> bool:
            nonlocal count
            if g["cap"] and cap_used[g["cap"]] >= CAPS[g["cap"]]:
                flags.append(
                    {"flag": "cap_reached", "set_id": g["rep"]["set_id"], "group_id": g["group_id"], "detail": g["cap"]}
                )
                return False
            if g["cap"]:
                cap_used[g["cap"]] += 1
            g["round"] = phase
            chosen_subs.add(g["rep"]["substances"])
            selected.append(g)
            count += 1
            return True

        # Hybrid (decision D11). Phase A: one slot per drug class (primary EPC, any route),
        # classes with more labels first; the class's largest group with a new substance set.
        by_class: dict[str, list] = defaultdict(list)
        for (klass, _route), b in buckets.items():
            if not klass.startswith("(no class)"):
                by_class[klass].extend(b)
        for klass in sorted(by_class, key=lambda k: (-sum(g["size"] for g in by_class[k]), k)):
            if count >= target:
                break
            for g in sorted(by_class[klass], key=lambda g: (-g["size"], g["group_id"])):
                if g["rep"]["substances"] not in chosen_subs and take(g, "class_slot"):
                    break
        # Phase B: fill the rest by popularity (group size), new substance sets first.
        taken = {g["group_id"] for g in selected}
        rest = sorted(
            (g for b in buckets.values() for g in b if g["group_id"] not in taken),
            key=lambda g: (-g["size"], g["group_id"]),
        )
        for allow_repeat in (False, True):
            for g in rest:
                if count >= target:
                    break
                if g["group_id"] in taken or (not allow_repeat and g["rep"]["substances"] in chosen_subs):
                    continue
                if take(g, "popularity"):
                    taken.add(g["group_id"])

    # 8. Candidate manifest. Fields not provided by these sources stay empty.
    candidates = []
    for g in selected:
        r = g["rep"]
        candidates.append(
            {
                "candidate_id": f"dailymed:{r['set_id']}",
                "source_name": "dailymed",
                "source_document_id": r["set_id"],
                "title": "",
                "source_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={r['set_id']}",
                "download_url": f"https://dailymed.nlm.nih.gov/dailymed/downloadzipfile.cfm?setId={r['set_id']}",
                "document_type": r["product_type"],
                "medicine_category": r["pharm_class_epc"],
                "medicine_category_source": "openfda.pharm_class_epc" if r["pharm_class_epc"] else "",
                "active_ingredient": r["substances"],
                "brand_name": r["brand_name"],
                "generic_name": r["generic_name"],
                "route": r["routes"],
                "dosage_form": r["dosage_forms"],
                "application_number": r["application_numbers"],
                "marketing_category": r["marketing_categories"],
                "is_brand": is_brand(r),
                "is_original_packager": r["is_original_packager"],
                "publisher": "U.S. National Library of Medicine (DailyMed)",
                "author_or_labeler": r["manufacturer_name"],
                "publication_date": "",
                "effective_time": r["effective_time"],
                "last_updated": "",
                "version": r["version"],
                "spl_id": r["spl_id"],
                "available_formats": "SPL XML (ZIP)|PDF",
                "license": "",
                "license_basis": "source_policy",
                "license_review_status": "pending",
                "collection_status": "candidate",
                "duplicate_group_id": g["group_id"],
                "group_member_count": g["size"],
                "completeness_score": r["completeness_score"],
                "has_boxed_warning": r["has_boxed_warning"],
                "excipients": r["excipients"],
                "cap_category": g["cap"],
                "selection_round": g["round"],
                "selection_reason": g["reason"],
                "rejection_reason": "",
                "discovered_via": f"openFDA bulk drug-label export {export_date} ({r['source_file']})",
                "discovered_at_utc": run_at,
            }
        )
    write_csv(args.out_dir / "candidates_dailymed.csv", candidates)
    write_csv(args.out_dir / "duplicate_groups.csv", member_rows)
    write_csv(args.out_dir / "review_flags.csv", flags, ["flag", "set_id", "group_id", "detail"])

    # Report.
    annotated = sum(lab["has_openfda"] for lab in labels)
    by_type = Counter(c["document_type"] for c in candidates)
    lines = [
        f"# DailyMed candidate selection — {run_at}",
        "",
        f"Source: openFDA bulk drug-label export {export_date}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Labels read | {len(labels)} |",
        f"| With openFDA metadata | {annotated} |",
        f"| Without openFDA metadata (unannotated pool) | {len(labels) - annotated} |",
        f"| Duplicate groups (Rx/OTC, complete key) | {len(groups)} |",
        f"| Candidates selected | {len(candidates)} |",
    ]
    lines += [f"| Candidates — {t} | {n} |" for t, n in by_type.items()]
    lines += [
        f"| Representative is a brand label | {sum(c['is_brand'] for c in candidates)} |",
        f"| Distinct substance sets | {len({c['active_ingredient'] for c in candidates})} |",
        f"| Distinct drug classes (EPC) | {len({x for c in candidates for x in c['medicine_category'].split('|') if x})} |",  # noqa: E501
        f"| Candidates without drug class | {sum(not c['medicine_category'] for c in candidates)} |",
        f"| Candidates with boxed warning | {sum(bool(c['has_boxed_warning']) for c in candidates)} |",
    ]
    lines += ["", "## Caps used", "", "| Category | Used | Cap |", "|---|---|---|"]
    lines += [f"| {k} | {cap_used[k]} | {v} |" for k, v in CAPS.items()]
    lines += ["", "## Review flags", "", "| Flag | Count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in Counter(f["flag"] for f in flags).most_common()]
    for title, field in (("Top routes", "route"), ("Top dosage forms", "dosage_form")):
        lines += ["", f"## {title}", "", "| Value | Candidates |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in Counter(c[field] for c in candidates).most_common(12)]
    (args.out_dir / "selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("done: %d candidates, %d flags -> %s", len(candidates), len(flags), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
