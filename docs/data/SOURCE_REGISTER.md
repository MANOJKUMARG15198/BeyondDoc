# Source Register — Phase 1

Status: **PROPOSED** — no source is approved yet (see ADR-009 in docs/DECISION_LOG.md).
Rule: a source moves to "Approved" only after a dated check of its official terms page, recorded below.

Last reviewed: 2026-11-03 · Reviewer: (pending)

## Decisions recorded (2026-11-03)

| ID | Decision | Status |
|---|---|---|
| D1 | `segment_id` is not used; dropped from the candidate manifest for now | Confirmed by user |
| D2 | Intended use is **non-commercial**. Source credits/attribution will be shown on a UI page | Confirmed by user |
| D3 | Raw documents stay **out of Git** for now. A later way to publish them (or a subset) is still to be decided | Confirmed by user; publishing method pending |
| D4 | Older label versions are **kept but not counted** toward the 1,000 target (for stale-source evaluation) | Confirmed by user |
| D5 | DailyMed discovery uses **option A**: stratify with openFDA `drug/label`, then confirm SETID/version with DailyMed | Confirmed by user |
| D6 | Representative label = **reference brand first** (NDA/BLA over ANDA). Tie-breakers, in order: (1) the application holder's own label over repackager labels, (2) most recent `effective_time`, (3) most complete set of sections | Confirmed by user |
| D6b | No brand label in a group → use the **most recent generic label** (by `effective_time`) | Confirmed by user |
| D6c | Duplicate group = **active ingredient(s) + route + dosage form + Rx/OTC** | Confirmed by user |
| D6d | Flag group members whose **inactive ingredients** differ in known allergens/excipients, for user review | Confirmed by user; excipient list pending |
| D9 | Sampling rules from the stage-1 inventory (2026-11-03, openFDA data of 2026-10-30): homeopathic products **small cap**; sunscreens and hand sanitizers **cap each at a few labels**; medical gases **one representative**; allergen extracts **a few representatives**; labels without openFDA annotations **also reached via DailyMed**; DailyMed split **~500 Rx / ~200 OTC**. Caps: homeopathic 10, sunscreens 5, hand sanitizers 5, medical gases 1, allergen extracts 5 — so that slots (and chunks) go to medically valuable content | Confirmed by user |
| D10 | Today's (2026-11-03) demo is an **end-to-end RAG** that answers questions. Approvals are **per stage**; all core architecture decisions (with scaling in mind) stay with the user | Confirmed by user |
| D11 | Candidate ranking = **hybrid**: one slot per drug class first (classes with more labels first), then fill by popularity (labels per group). Result 2026-11-03: 910 candidates (650 Rx / 260 OTC), 452 drug classes, 874 substance sets, 431 brand representatives | Confirmed by user |
| D8 | Python version, project/dependency tooling and `.gitignore` are set in the foundation phase. Phase 1 discovery scripts use the Python standard library only, run with the installed Python, and install nothing | Confirmed by user |
| D7 | Proposed allocation 700 DailyMed / 100 FDA / 150 PLOS+PMC / 50 WHO-or-NIH | Proposed, not approved |

## Sources

Verification key: **V** = checked on the official page on 2026-11-03 · **U** = not yet verified (page blocked or not read).

| Source | Intended use | Access method | Official docs | Rate limits | Formats | License / reuse terms | Non-commercial use | Status |
|---|---|---|---|---|---|---|---|---|
| DailyMed (NLM) | Rx/OTC labels (current + older versions) | Web services v2: `/spls.json` (filters incl. `doctype`, `setid`, `unii_code`, `rxcui`, `drug_class_code`, `published_date`; `pagesize` max 100), `/spls/{SETID}/history`; files via `downloadzipfile.cfm?setId=`, `downloadpdffile.cfm?setId=`; older versions ZIP only via `getFile.cfm?type=zip&setid=&version=` **V** | https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm | Not published **V** — self-limit | SPL XML (in ZIP), PDF, ZIP with images | No terms on the DailyMed page **V**. NLM: US-government works not copyrighted; privately contributed material may be copyrighted; credit "Source: National Library of Medicine" requested **V**. Label text is manufacturer-submitted — status **U** | Likely OK — **U** | Proposed |
| openFDA | Metadata enrichment only (stratification) | REST `drug/label` | https://open.fda.gov/apis/authentication/ , https://open.fda.gov/terms/ | 240/min; 1,000/day without key, 120,000/day with key **V** | JSON | Public domain, CC0 1.0, unless otherwise noted; credit to FDA requested; GMDN exception (devices, not used) **V** | Yes **V** | Proposed |
| Drugs@FDA | Regulatory documents | Daily data file (12 tab-delimited tables incl. `ApplicationDocs`: ApplNo, SubmissionType/No, Title, URL, Date) **V**; document PDFs from the listed URLs | https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files | File updated Mon–Fri **V**; PDFs: self-limit | PDF | US-government work; third-party content possible — **U** | Likely OK — **U** | Proposed |
| PLOS | Open-access pharma research | Solr Search API (`rows` ≤ 100) **V**; article XML/PDF URLs | https://api.plos.org/solr/faq/ , https://journals.plos.org/plosone/s/licenses-and-copyright | 10/min, 300/hour, 7,200/day, 5 concurrent connections per IP; IP lockout if exceeded **V** | JATS XML, PDF | CC BY 4.0 (some CC0); credit PLOS as data source; logo needs permission **V** | Yes **V** | Proposed |
| PMC Open Access | Open-access pharma research | **OA Web Service and FTP retired Aug 2026** **V**. Use E-utilities `esearch` with license filters for discovery, then the AWS Open Data bucket `pmc-oa-opendata` (anonymous HTTPS, per-version prefix `PMCxxxx.N/` with `.xml`, `.txt`, `.json`, `.pdf` when permitted) **V** | https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/ , https://pmc-oa-opendata.s3.amazonaws.com/README.txt | E-utilities: 3 req/s without key, 10 with key (per NLM support article; NCBI book page blocked) | JATS XML, TXT, PDF (when permitted), JSON metadata | Per article: `license_code` in metadata JSON; also `is_retracted`, `doi`, `pmid`, `version` **V** | Per article (CC BY / CC BY-NC both usable under D2) | Proposed |
| WHO IRIS | Public-health reference | DSpace REST / OAI-PMH — **U** (iris.who.int returned 403 to the checker) | https://www.who.int/about/policies/publishing/copyright | **U** | PDF | CC BY-NC-SA 3.0 IGO; required citation format; adaptations need disclaimer + same licence; figures/tables/photos may need permission; no logo, no endorsement **V** | Yes **V** | Proposed |
| NIH — LiverTox | Drug-induced liver injury reference | NCBI Bookshelf / E-utilities — **U** | https://www.ncbi.nlm.nih.gov/books/NBK547852/ | E-utilities limits | HTML, PDF | Described as public domain; acknowledgment requested — from search results only, **U** | Likely OK — **U** | Proposed |
| NIH — ODS fact sheets | Dietary supplement reference | Web pages — **U** | https://ods.od.nih.gov/ (copyright page returned 403) | **U** | HTML | **U** | **U** | Proposed |

## Attribution (for the UI credits page, per D2)
- NLM / DailyMed: "Source: National Library of Medicine".
- openFDA / Drugs@FDA: data provided by the U.S. Food and Drug Administration, with a link to open.fda.gov.
- PLOS: credit PLOS; cite each article (authors, title, journal, DOI) — CC BY 4.0.
- PMC: cite each article and show its `license_code`.
- WHO: "[Title]. [Place]: World Health Organization; [Year]. Licence: CC BY-NC-SA 3.0 IGO." No WHO logo, no implied endorsement.

## Redistribution notes (for D3)
- CC BY-NC-SA material (WHO, some PMC) can be redistributed only non-commercially, with attribution and under the same licence.
- Figures, tables and photos inside WHO and some article packages can carry separate rights; prefer redistributing text-only derivatives or manifests + fetch scripts.

## Explicit exclusions
- MedlinePlus drug monographs (ASHP/AHFS licensed) and A.D.A.M. encyclopedia content.
- Patient records, personal health information, diagnosis datasets, identifiable case reports.
- Retracted articles.

## Change log
- 2026-11-03: Created as proposal; decisions D1–D5 recorded. No sources approved.
- 2026-11-03: Official docs/terms checked (see V/U markers). Found that the PMC OA Web Service was retired in Aug 2026; access plan changed to E-utilities + AWS bucket.
- 2026-11-03: D6 set to reference-brand-first (R1); D6–D6d and D8 recorded.
- 2026-11-03: Stage-1 openFDA inventory run (13 count queries, 0 failures); D9 recorded.
