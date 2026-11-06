"""Retrieval v1.2 Step 5: new intents, single-study rule, section boosts and noise demotion."""

from medical_rag.query.intent import detect_intents, refers_to_single_study
from medical_rag.query.medicine import MedicineResolution, ResolvedMedicine
from medical_rag.search.planner import STANDARD, coverage_units, make_plan
from medical_rag.search.retrieval import bm25_body, section_boosts
from medical_rag.search.selection import apply_noise_demotion, is_noise


def res(*names):
    found = [ResolvedMedicine(n, n, (0, 1), "verified", (f"dailymed:{n}", f"pmc_oa:{n}")) for n in names]
    return MedicineResolution(found, [], len(found) > 1, "verified" if found else "none", "v")


def test_new_intents():
    assert "indications" in detect_intents("What is Lyrica approved for?")
    assert "limitations" in detect_intents("What limitations did the trial report?")
    assert "overdose" in detect_intents("What happened when adults received much higher doses?")
    assert "overdose" in detect_intents("What are the signs of overdosage?")


def test_single_study_rule_skips_per_medicine_units_only():
    assert refers_to_single_study("Did trial sequential analysis confirm celecoxib beats ibuprofen?")
    assert not refers_to_single_study("What do the two labels say about kidney function?")
    units, notes = coverage_units(res("a", "b"), [], single_study=True)
    assert units == [] and notes == ["per-medicine units skipped: single study"]
    assert len(coverage_units(res("a", "b"), [], single_study=False)[0]) == 2
    named, _ = coverage_units(res("a"), ["dailymed", "pmc_oa"], single_study=True)
    assert len(named) == 2  # named-source units are unaffected


def test_sections_map_intents_and_apply_without_a_medicine():
    plan = make_plan(res(), ["dosage", "case_report"], sections=True)
    assert plan.name == "section" and plan.section_phrases == ("dosage and administration",)
    assert plan.section_roles == ("case_report",) and not plan.is_standard
    assert make_plan(res(), ["dosage"]) is STANDARD  # off unless enabled


def test_noise_demotion_is_exempt_for_bibliography_questions():
    assert make_plan(res("a"), [], noise=True).demote_noise
    assert not make_plan(res("a"), ["bibliography"], noise=True).demote_noise


def test_section_boosts_are_soft_should_clauses():
    plan = make_plan(res(), ["pregnancy_lactation", "case_report"], sections=True, section_boost=2.0)
    body = bm25_body("q", 5, None, section_boosts(plan))
    should = body["query"]["bool"]["should"]
    assert {"match_phrase": {"section_path": {"query": "pregnancy", "boost": 2.0}}} in should
    assert {"term": {"section_role": {"value": "case_report", "boost": 2.0}}} in should
    assert body["query"]["bool"]["must"] == [{"match": {"search_text": "q"}}]
    assert "should" not in bm25_body("q", 5, None, section_boosts(STANDARD))["query"]["bool"]


class H:
    def __init__(self, cid, **source):
        self.chunk_id, self.source = cid, source


def test_noise_detection():
    assert is_noise({"section_role": "bibliography"}) and is_noise({"section_role": "letter_signature"})
    assert is_noise({"source": "dailymed", "section_role": "body", "section_path": "SPL MEDGUIDE SECTION (part 3)"})
    assert not is_noise({"source": "dailymed", "section_role": "body", "section_path": "17 PATIENT COUNSELING"})
    assert not is_noise({"source": "pmc_oa", "section_role": "body", "section_path": "Patient information"})


def test_noise_demotion_keeps_one_and_never_deletes():
    bib = {"section_role": "bibliography"}
    hits = [H("b1", **bib), H("x1"), H("b2", **bib), H("b3", **bib), H("x2")]
    out, reasons = apply_noise_demotion(hits, max_noise=1)
    assert [h.chunk_id for h in out] == ["b1", "x1", "x2", "b2", "b3"]
    assert reasons == {"b2": "demoted_noise", "b3": "demoted_noise"}


def test_settings_plan_is_off_by_default_and_builds_t3_when_enabled(tmp_path):
    from medical_rag.core.config import Settings
    from medical_rag.query.medicine import AliasTable
    from medical_rag.search.planning import plan_for

    assert plan_for("gabapentin dose", Settings(_env_file=None)) is None
    path = tmp_path / "aliases_x.json"
    path.write_text(AliasTable(aliases={}, documents={}, version="x").to_json(), encoding="utf-8")
    on = Settings(_env_file=None, retrieval_plan="v12_t3", alias_table_path=str(path))
    plan = plan_for("What dose is recommended? See the references.", on)
    assert plan is not None and plan.section_phrases == ("dosage and administration",) and not plan.demote_noise
    assert plan_for("q", on, mode="bm25") is None
