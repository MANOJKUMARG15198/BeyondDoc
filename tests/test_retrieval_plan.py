"""Retrieval v1.2 planner, identity guard and plan-aware search (fake OpenSearch client, no network)."""

import numpy as np
import pytest

from medical_rag.query.medicine import MedicineResolution, ResolvedMedicine
from medical_rag.search.planner import STANDARD, make_plan
from medical_rag.search.retrieval import Hit, Retriever, bm25_body
from medical_rag.search.selection import apply_identity_guard


def res(*meds, confidence="verified", comparison=False):
    found = [ResolvedMedicine(m, m, (0, len(m)), "verified", docs) for m, docs in meds]
    return MedicineResolution(found, [], comparison or len(found) > 1, confidence, "v")


IBU = ("ibuprofen", ("dm:ibu", "pmc:ibu"))


def test_plan_fallbacks():
    assert make_plan(MedicineResolution()) is STANDARD and STANDARD.is_standard
    derived = make_plan(res(IBU, confidence="derived"))
    assert derived.boost_documents and not derived.medicine_dense_documents and derived.guard is None
    verified = make_plan(res(IBU))
    assert verified.medicine_dense_documents == ("dm:ibu", "pmc:ibu") and verified.guard.max_other == 2
    assert make_plan(res(IBU), ["interactions"]).guard is None
    assert make_plan(res(IBU, ("naproxen", ("dm:nap",)))).guard is None  # two medicines: comparison
    assert make_plan(res(IBU), filtered_dense=False, guard=False).medicine_dense_documents == ()


class H:
    def __init__(self, cid, doc):
        self.chunk_id, self.source = cid, {"document_id": doc}


def test_guard_demotes_never_deletes_and_keeps_order():
    hits = [H("a", "x"), H("b", "dm:ibu"), H("c", "y"), H("d", "z"), H("e", "dm:ibu"), H("f", "w")]
    out, reasons = apply_identity_guard(hits, frozenset({"dm:ibu"}), max_other=2)
    assert [h.chunk_id for h in out] == ["a", "b", "c", "e", "d", "f"]
    assert reasons == {"d": "demoted_other_medicine", "f": "demoted_other_medicine"} and len(out) == len(hits)


def test_guard_still_fills_k_when_the_medicine_has_few_passages():
    hits = [H(c, "other") for c in "abcd"]
    out, _ = apply_identity_guard(hits, frozenset({"dm:ibu"}), max_other=1)
    assert [h.chunk_id for h in out] == list("abcd")


def test_bm25_body_without_boosts_is_unchanged():
    body = bm25_body("dose", 5, {"source": "dailymed"})
    assert body["query"]["bool"] == {"must": [{"match": {"search_text": "dose"}}],
                                     "filter": [{"terms": {"source": ["dailymed"]}}]}  # fmt: skip


class FakeClient:
    def __init__(self, results):
        self.results, self.bodies = results, []

    def search(self, index, body):
        self.bodies.append(body)
        if "knn" in body["query"]:
            name = "dense_medicine" if "filter" in body["query"]["knn"]["embedding"] else "dense"
        else:
            name = "bm25"
        return {"hits": {"hits": [{"_id": c, "_score": 1.0, "_source": {"chunk_id": c, "document_id": d}}
                                  for c, d in self.results[name]]}}  # fmt: skip


class FakeEmbedder:
    def embed_query(self, text):
        return np.ones(2)


RESULTS = {
    "bm25": [("n1", "dm:nap"), ("i1", "dm:ibu"), ("n2", "dm:nap"), ("n3", "dm:nap")],
    "dense": [("n1", "dm:nap"), ("n2", "dm:nap"), ("i2", "dm:ibu"), ("n3", "dm:nap")],
    "dense_medicine": [("i2", "dm:ibu"), ("i1", "dm:ibu")],
}


def test_search_without_plan_is_identical_to_the_standard_plan():
    a = Retriever(FakeClient(RESULTS), FakeEmbedder()).search("q", k=3)
    b = Retriever(FakeClient(RESULTS), FakeEmbedder()).search("q", k=3, plan=STANDARD)
    assert [h.chunk_id for h in a.hits] == [h.chunk_id for h in b.hits] and a.selection == b.selection == {}


def test_medicine_plan_adds_boosts_a_filtered_dense_list_and_the_guard():
    client = FakeClient(RESULTS)
    plan = make_plan(res(("ibuprofen", ("dm:ibu",))), max_other=1)
    out = Retriever(client, FakeEmbedder()).search("ibuprofen pregnancy", k=3, plan=plan)
    bm25 = client.bodies[0]["query"]["bool"]["should"]
    assert {"terms": {"document_id": ["dm:ibu"], "boost": 2.0}} in bm25
    knn = client.bodies[2]["query"]["knn"]["embedding"]
    assert knn["filter"]["bool"]["filter"] == [{"terms": {"document_id": ["dm:ibu"]}}]
    # RRF over 3 lists: n1 2/61, i2 1/61+1/63, i1 2/62, n2 1/62+1/63, n3 2/64; the guard (max 1 other) demotes n2, n3
    assert [h.chunk_id for h in out.hits] == ["n1", "i2", "i1"]
    assert out.hits[1].ranks == {"dense": 3, "dense_medicine": 1}
    assert "dense_medicine_ms" in out.timings_ms and out.plan_reasons


def test_plan_requires_hybrid_mode():
    with pytest.raises(ValueError):
        Retriever(FakeClient(RESULTS), FakeEmbedder()).search("q", mode="bm25", plan=make_plan(res(IBU)))


def test_hit_type_is_unchanged():
    assert Hit("c", 1.0, {}).ranks == {}
