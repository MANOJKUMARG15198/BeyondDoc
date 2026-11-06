"""Retrieval v1.2 Step 4: named-source detection, coverage units, coverage floor and branch execution (fake client)."""

import numpy as np

from medical_rag.query.intent import coverage_sources
from medical_rag.query.medicine import MedicineResolution, ResolvedMedicine
from medical_rag.search.planner import MAX_UNITS, coverage_units, make_plan
from medical_rag.search.retrieval import Retriever
from medical_rag.search.selection import apply_coverage_floor

GABA = ("gabapentin", ("dailymed:gaba", "nih_livertox:gaba", "pmc_oa:gaba"))


def res(*meds, confidence="verified"):
    found = [ResolvedMedicine(m, m, (0, 1), "verified", docs) for m, docs in meds]
    return MedicineResolution(found, [], len(found) > 1, confidence, "v")


def test_secondary_research_cues_count_only_next_to_a_named_source():
    assert coverage_sources("Compare the gabapentin label with LiverTox") == ["dailymed", "nih_livertox"]
    assert coverage_sources("What does the label say, and what did the ICU trial find?") == ["dailymed", "pmc_oa"]
    assert coverage_sources("What did the study find about gabapentin?") == []
    assert coverage_sources("What does the label say about dosing?") == ["dailymed"]


def test_units_per_named_source_and_unavailable_sources_are_logged():
    units, notes = coverage_units(res(GABA), ["dailymed", "nih_livertox", "fda_drugsatfda"])
    assert [u.key for u in units] == ["gabapentin/dailymed", "gabapentin/nih_livertox"]
    assert units[0].documents == ("dailymed:gaba",) and notes == ["source_unavailable:gabapentin/fda_drugsatfda"]


def test_units_per_medicine_for_two_medicines_and_none_for_single_source_or_unverified():
    two = coverage_units(res(("a", ("dailymed:a",)), ("b", ("dailymed:b",))), [])[0]
    assert [u.key for u in two] == ["a/any", "b/any"]
    assert coverage_units(res(GABA), ["dailymed"])[0] == []
    assert coverage_units(res(GABA, confidence="derived"), ["dailymed", "pmc_oa"])[0] == []


def test_units_are_capped():
    meds = [(f"m{i}", (f"dailymed:{i}", f"pmc_oa:{i}")) for i in range(3)]
    units, notes = coverage_units(res(*meds), ["dailymed", "pmc_oa"])
    assert len(units) == MAX_UNITS and notes


def test_coverage_is_off_unless_enabled():
    assert make_plan(res(GABA), sources=["dailymed", "nih_livertox"]).coverage == ()
    plan = make_plan(res(GABA), sources=["dailymed", "nih_livertox"], coverage=True, coverage_floor=2)
    assert len(plan.coverage) == 2 and plan.coverage_floor == 2 and not plan.is_standard


class H:
    def __init__(self, cid):
        self.chunk_id, self.source = cid, {"document_id": "d"}


def test_floor_reserves_lexically_matched_branch_passages_and_keeps_combined_order():
    hits = [H(c) for c in ["m1", "m2", "m3", "m4", "m5", "m6", "x1", "x2", "y1"]]
    units = [("x", ["x2", "x1"], {"x1", "x2"}), ("y", ["y1", "y9"], {"y9"})]  # y1 has no BM25 match
    final, reasons = apply_coverage_floor(hits, units, floor=2, k=6)
    assert [h.chunk_id for h in final] == ["m1", "m2", "m3", "m4", "x1", "x2"]
    assert reasons["x2"] == reasons["x1"] == "coverage:x" and reasons["m1"] == "relevance"


def test_floor_is_capped_at_half_of_k_and_shared_round_robin():
    hits = [H(c) for c in ["m1", "m2", "a1", "a2", "a3", "b1", "b2", "b3"]]
    units = [("a", ["a1", "a2", "a3"], {"a1", "a2", "a3"}), ("b", ["b1", "b2", "b3"], {"b1", "b2", "b3"})]
    _, reasons = apply_coverage_floor(hits, units, floor=3, k=4)
    covered = sorted(c for c, r in reasons.items() if r.startswith("coverage"))
    assert covered == ["a1", "b1"]  # cap 2 = k // 2, one per unit first


class FakeClient:
    """Answers by query type and document filter; can fail branch queries."""

    def __init__(self, fail_branches=False):
        self.fail_branches, self.bodies = fail_branches, []

    def search(self, index, body):
        self.bodies.append(body)
        query = body["query"]
        flt = query["knn"]["embedding"].get("filter", {}).get("bool", {}).get("filter", []) if "knn" in query else (
            query["bool"]["filter"])  # fmt: skip
        docs = next((f["terms"]["document_id"] for f in flt if "document_id" in f.get("terms", {})), None)
        if docs == ["nih_livertox:gaba"]:
            if self.fail_branches:
                raise TimeoutError("branch")
            ids = [("lt1", "nih_livertox:gaba"), ("lt2", "nih_livertox:gaba")]
        elif docs == ["dailymed:gaba"]:
            ids = [("dm9", "dailymed:gaba")]
        else:
            ids = [(f"dm{i}", "dailymed:gaba") for i in range(1, 9)]
        return {"hits": {"hits": [{"_id": c, "_score": 1.0, "_source": {"chunk_id": c, "document_id": d}}
                                  for c, d in ids]}}  # fmt: skip


class FakeEmbedder:
    def embed_query(self, text):
        return np.ones(2)


def plan(**kw):
    return make_plan(res(GABA), sources=["dailymed", "nih_livertox"], coverage=True, coverage_floor=2, **kw)


def test_branches_bring_the_second_source_into_the_final_8():
    out = Retriever(FakeClient(), FakeEmbedder()).search("gabapentin label and LiverTox", k=8, plan=plan())
    ids = [h.chunk_id for h in out.hits]
    assert {"lt1", "lt2"} <= set(ids) and len(ids) == 8
    assert out.selection["lt1"] == "coverage:gabapentin/nih_livertox" and "coverage_ms" in out.timings_ms


def test_failed_branch_is_skipped_not_fatal():
    out = Retriever(FakeClient(fail_branches=True), FakeEmbedder()).search("q", k=8, plan=plan())
    assert len(out.hits) == 8 and "branch_failed:gabapentin/nih_livertox" in out.selection
    assert not any(h.chunk_id.startswith("lt") for h in out.hits)


def test_plan_without_units_is_identical_to_the_m2_plan():
    m2 = make_plan(res(GABA), sources=["dailymed"])
    with_flag = make_plan(res(GABA), sources=["dailymed"], coverage=True)
    a = Retriever(FakeClient(), FakeEmbedder()).search("q", k=8, plan=m2)
    b = Retriever(FakeClient(), FakeEmbedder()).search("q", k=8, plan=with_flag)
    assert [h.chunk_id for h in a.hits] == [h.chunk_id for h in b.hits] and with_flag.coverage == ()
