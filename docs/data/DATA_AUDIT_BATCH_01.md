# Data Quality Audit — Batch 1

**Date:** 2026-11-03 · **Scope:** the 100 documents collected in Batch 1 · **Method:** read-only inspection (commands C1–C10); no files were changed or extracted to disk.

## 1. Summary

- All **100 documents / 181 files** are present, unchanged and readable.
- **XML is the cleaner text source** for DailyMed, PMC and LiverTox. FDA letters exist **only as PDF**.
- **No exact duplicates.** 11 drug-label pairs share **FDA class-labeling text** (opioid combinations, NSAIDs), which is a retrieval risk.
- Metadata gaps can all be **filled from the documents themselves**.
- Four content questions need a user decision (section 7).

## 2. Counts and integrity

| Check | Result |
|---|---|
| Logical documents | 100 — DailyMed 70 · FDA 10 · PMC 15 · LiverTox 5 |
| Duplicate document IDs | None |
| Files | 181 unique — DailyMed 70 ZIP + 70 PDF · FDA 10 PDF · PMC 15 XML + 15 PDF · LiverTox 1 archive (3,439 members) |
| SHA-256 and size vs manifest | 181 / 181 match |
| Archive tests | 70 / 70 ZIPs pass; LiverTox tar.gz opens |
| DailyMed document types | 50 Human Prescription Drug Label · 20 Human OTC Drug Label (matches the Rx/OTC plan) |

## 3. Document structure

| | DailyMed (SPL XML) | PMC (JATS XML) | LiverTox (NXML) | FDA letters (PDF) |
|---|---|---|---|---|
| Size | 142 – 23,351 words (median 7,686; total 587k) | 2,247 – 11,228 words (total 69k) | 1,213 – 14,275 words (total 23k) | 3 – 5 pages (~11k words total) |
| Hierarchy | 1,242 top-level / 3,389 total sections; depth up to 5 | 3 – 42 sections; depth ≤ 3 | Fixed chapter template, 8 – 28 sections | None |
| Section labels | LOINC section codes (Indications, Dosage, Warnings …) | `<sec><title>` | Identical headings in every chapter | — |
| Tables | 427 (median 3; max 32; 20 labels have none) | 60 tables · 43 figures | 1 – 9 | — |
| Other | 356 package-label images · 1,338 lists | Abstract in all 15 · 714 references | 9 – 159 references | Letterhead, signatures |

**Examples:**
- **Synthroid** (`dailymed/1e11ad30-…`): clean, numbered sections in the standard FDA label format (5.1 – 5.6, 7.1 – 7.10); 4 tables in *7.1 Drug Interactions*. Some headings have inconsistent numbering ("5. 2").
- **Abilify** (`dailymed/c040bd1d-…`): the hardest label. It has 187 sections, **119 untitled subsections**, and 5,426 words in *5 Warnings and Precautions* alone.
- **OTC Drug Facts** labels (e.g. `dailymed/5d74f3a2-…`): very short sections of 3 – 33 words (Active ingredient, Purpose, Uses, Warnings).
- **PMC flat reviews:** 4 of the 7 reviews use only *Introduction / Review / Conclusions*, so one section can be **9,680 words** (`PMC13570720.1`).
- **LiverTox:** every chapter follows Overview → Hepatotoxicity → Mechanism → Outcome → Case reports → Product information → Annotated bibliography. In *Atorvastatin*, the bibliography (159 references) is a large share of the text.

## 4. Text quality (XML vs PDF)

| Source | XML tokens found in PDF | PDF tokens found in XML | PDF problems |
|---|---|---|---|
| DailyMed | 99.9 % | 93.9 % (small OTC labels: 29 – 42 %) | 27 empty (image-only) pages, repeated headers/footers, 1,223 broken hyphenations, flattened tables |
| PMC | 92.4 % | 95.3 % (`PMC3787571.1`: 68 %) | 676 NUL characters in `PMC13004290.1`, 255 ligatures, 673 hyphen breaks, journal headers |
| LiverTox | — | — | Clean |
| FDA letters | — | — | Clean; letterhead and boilerplate text |

**Observation:** the XML holds the complete, structured text, without the page artifacts of the PDFs. The PDFs are still needed for FDA letters, and are useful for page-level citations. *Which format is primary is a pending decision.*

## 5. Duplicates and overlap

- **Exact duplicates:** none, either whole documents or top-level label sections (> 30 words).
- **Near-duplicates (shared 8-word passages):** 11 label pairs, all caused by **FDA class labeling**:

| Pair | Shared |
|---|---|
| Hydrocodone/APAP ↔ Codeine/APAP | 34 % |
| Hydrocodone/APAP ↔ Oxycodone/APAP | 33 % |
| Celebrex ↔ Diclofenac | 27 % |
| Naproxen ↔ Diclofenac | 26 % |
| Meloxicam ↔ Naproxen | 26 % |

  **Risk:** a question about one drug may retrieve identical warning text from another drug's label. Chunks need drug and label identity attached. These passages must **not** be deduplicated away, because each label legally carries its own copy.
- **Same drug across sources:** 19 drugs appear in more than one source. **Atorvastatin and pregabalin appear in all four**, which makes them good multi-source evaluation cases.

## 6. Metadata gaps

| Gap | Where the value is available |
|---|---|
| DailyMed official title and publication date | SPL `<title>` and `effectiveTime` |
| PMC publication date and version | JATS `<pub-date>`; version is already in the ID (`.1`) |
| LiverTox last-update date | NXML metadata |
| License (DailyMed, FDA, LiverTox) | Intentionally at source-policy level; review pending |

## 7. Questions for decision

1. **Label boilerplate:** 172 package-label display panels and 54 product-data sections, mostly NDC codes and packaging text. Index them, or exclude them?
2. **Case reports:** LiverTox chapters include published, de-identified case reports, but PMC case reports were excluded. Decide one rule for both.
3. **Names in FDA letters:** company contacts and FDA signatories (professionals, not patients). Keep them, or redact them?
4. **Corrected article:** `PMC13004290.1` is a corrected version, with the correction notice in its PDF. Proposed: keep it and note the correction.

## 8. Representative 20-document sample (for the chunking benchmark)

| # | Source | Document | Reason |
|---|---|---|---|
| 1 | DailyMed Rx | Synthroid (45 pp) | Standard label format, tables |
| 2 | DailyMed Rx | Prednisone (20 pp) | No tables |
| 3 | DailyMed Rx | Lipitor (43 pp) | Multi-source drug |
| 4 | DailyMed Rx | Metformin (28 pp) | Typical size |
| 5 | DailyMed boxed warning | Abilify (77 pp) | Deepest hierarchy, untitled subsections |
| 6 | DailyMed boxed warning | Oxycodone/APAP (36 pp) | Opioid class text |
| 7 | DailyMed boxed warning | Diclofenac (28 pp) | NSAID class text |
| 8 | DailyMed OTC | SurviveX (8 pp) | Shortest label (142 words) |
| 9 | DailyMed OTC | Claritin (13 pp) | Drug Facts format |
| 10 | DailyMed OTC | Mucinex (5 pp) | Drug Facts format |
| 11 | FDA | Synthroid original approval letter | PDF only |
| 12 | FDA | Lipitor supplement letter | PDF only |
| 13 | FDA | Prozac supplement letter | PDF only |
| 14 | PMC | Allopurinol systematic review (`PMC13004290.1`) | NUL characters, corrected article |
| 15 | PMC | Cyclobenzaprine PK study (`PMC3787571.1`) | Lowest XML/PDF agreement |
| 16 | PMC | Rapamycin/prednisone review (`PMC13509944.1`) | Nested sections |
| 17 | PMC | Pregabalin trial (`PMC13529079.1`) | Research-article structure |
| 18 | LiverTox | Cyclobenzaprine | Short chapter |
| 19 | LiverTox | Gabapentin | Medium chapter |
| 20 | LiverTox | Atorvastatin | Long chapter, case reports |

## 9. Not covered by this audit

- Medical correctness of the content (all sources are official publishers).
- Chunking — D-2 Steps 2 and 3, still pending.
- Batches 2 – 4.
