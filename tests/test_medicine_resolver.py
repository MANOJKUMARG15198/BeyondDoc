"""Deterministic medicine resolver: corpus-only aliases, normalization, confidence and fallbacks (in-memory chunks)."""

from medical_rag.query.intent import detect_intents, detect_sources
from medical_rag.query.medicine import AliasTable, build_alias_table, resolve_medicines


def chunk(doc, source, names, title="t", section=("Body",), text=""):
    return {"document_id": doc, "source": source, "medicine_names": names, "title": title,
            "section_path": list(section), "text": text}  # fmt: skip


CHUNKS = [
    chunk("dm:lipitor", "dailymed", ["Lipitor", "atorvastatin calcium"]),
    chunk("fda:lipitor", "fda_drugsatfda", ["Lipitor", "atorvastatin"]),
    chunk("lt:atorva", "nih_livertox", ["Atorvastatin"], section=("PRODUCT INFORMATION",),
          text="REPRESENTATIVE TRADE NAMES\nAtorvastatin – Generic, Lipitor®\nDRUG CLASS\nAntilipemic"),
    chunk("lt:preg", "nih_livertox", ["Pregabalin"], section=("PRODUCT INFORMATION",),
          text="REPRESENTATIVE TRADE NAMES\nPregabalin – Generic, Lyrica®\nDRUG CLASS\nAnticonvulsants"),
    chunk("dm:ibu", "dailymed", ["Ibuprofen"]),
    chunk("dm:naproxen", "dailymed", ["Naproxen"]),
    chunk("dm:hydro", "dailymed", ["hydrocodone bitartrate and acetaminophen"]),
    chunk("dm:codeine", "dailymed", ["Acetaminophen and Codeine Phosphate"]),
    chunk("dm:cozaar", "dailymed", ["COZAAR", "losartan potassium"]),
    chunk("dm:celebrex", "dailymed", ["CELEBREX", "Celecoxib"]),
    chunk("dm:hs", "dailymed", ["Head and Shoulders Bare Minimal Ingredients", "Pyrithione Zinc"]),
    chunk("dm:degree", "dailymed", ["Degree", "Advanced Protection Antiperspirant"]),
    chunk("dm:warfarin", "dailymed", ["Warfarin Sodium", "Warfarin"]),
]  # fmt: skip
TABLE = build_alias_table(CHUNKS)


def test_brand_and_generic_resolve_to_one_medicine_with_all_its_documents():
    brand, generic = resolve_medicines("Lipitor dose?", TABLE), resolve_medicines("atorvastatin dose?", TABLE)
    assert [m.medicine_id for m in brand.medicines] == ["atorvastatin"] == [m.medicine_id for m in generic.medicines]
    assert brand.document_ids() == ["dm:lipitor", "fda:lipitor", "lt:atorva"] == generic.document_ids()
    assert brand.confidence == "verified"


def test_livertox_trade_name_link_comes_from_the_document():
    r = resolve_medicines("Can Lyrica injure the liver?", TABLE)
    assert [m.medicine_id for m in r.medicines] == ["pregabalin"] and r.document_ids() == ["lt:preg"]


def test_salt_form_is_derived_and_exact_stored_name_is_verified():
    assert resolve_medicines("losartan potassium tablets", TABLE).confidence == "verified"
    assert resolve_medicines("Is losartan safe?", TABLE).confidence == "derived"


def test_longest_match_and_combination_components():
    combo = resolve_medicines("hydrocodone bitartrate and acetaminophen overdose", TABLE)
    assert [m.medicine_id for m in combo.medicines] == ["acetaminophen+hydrocodone"]
    assert resolve_medicines("Is acetaminophen safe?", TABLE).confidence == "ambiguous"  # two combination products


def test_product_names_are_never_split():
    assert "head" not in TABLE.aliases and resolve_medicines("my head hurts", TABLE).medicines == []


def test_two_medicines_make_a_comparison():
    r = resolve_medicines("Can Cozaar be taken with Celebrex?", TABLE)
    assert {m.medicine_id for m in r.medicines} == {"losartan", "celecoxib"} and r.comparison


def test_guarded_common_word_brand_needs_stored_capitalisation():
    table = build_alias_table(CHUNKS, [{"action": "guard", "alias": "degree", "guard_form": "Degree",
                                        "status": "approved"}])  # fmt: skip
    assert resolve_medicines("Does Degree contain aluminum?", table).medicines
    lower = resolve_medicines("to what degree does it help?", table)
    assert lower.medicines == [] and lower.confidence == "ambiguous"
    assert resolve_medicines("warfarin bleeding risk", table).confidence == "verified"  # generics never guarded


def test_unapproved_overlay_rows_are_ignored_and_approved_ones_applied():
    rows = [{"action": "add_alias", "alias": "Pepcid AC Original Strength", "medicine_id": "famotidine",
             "document_id": "dm:pepcid", "status": "proposed"}]  # fmt: skip
    assert "pepcid ac original strength" not in build_alias_table(CHUNKS, rows).aliases
    rows[0]["status"] = "approved"
    table = build_alias_table(CHUNKS, rows)
    assert resolve_medicines("Pepcid AC Original Strength ingredients", table).document_ids() == ["dm:pepcid"]


def test_no_medicine_misspelling_and_unknown_drug_fall_back_to_none():
    for q in ("What did the IP-10 study conclude?", "atorvastin dose", "dupilumab loading dose"):
        assert resolve_medicines(q, TABLE).confidence == "none"


def test_no_invented_brand_generic_links():
    assert resolve_medicines("Advil dose", TABLE).confidence == "none"  # Advil appears in no corpus record


def test_table_round_trip_and_version_are_deterministic():
    again = build_alias_table(list(reversed(CHUNKS)))
    assert again.version == TABLE.version and AliasTable.from_json(TABLE.to_json()).version == TABLE.version
    assert resolve_medicines("Lipitor", TABLE).version == TABLE.version


def test_intents_and_named_sources():
    assert detect_intents("Can I take ibuprofen while pregnant?") == ["pregnancy_lactation"]
    assert "case_report" in detect_intents("In the LiverTox case 3, which drugs?")
    assert "bibliography" in detect_intents("Which references discuss atorvastatin?")
    assert detect_intents("Tell me about it") == []
    assert detect_sources("Compare the gabapentin label with LiverTox") == ["dailymed", "nih_livertox"]
