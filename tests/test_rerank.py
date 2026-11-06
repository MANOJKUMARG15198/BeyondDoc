"""Reranking with a fake scorer and a whitespace token counter (no model download)."""

import importlib.util
from pathlib import Path

import pytest

from medical_rag.search.rerank import make_windows, rerank

ROOT = Path(__file__).resolve().parents[1]


def count(text: str) -> int:
    return len(text.split())


class KeywordScorer:
    """Score = number of times the keyword appears in the passage; records every pair it saw."""

    def __init__(self, keyword: str) -> None:
        self.keyword, self.seen = keyword, []

    def predict(self, pairs, batch_size):
        self.seen.extend(pairs)
        return [p.lower().count(self.keyword) for _, p in pairs]


def test_short_passage_is_one_window():
    assert [w.text for w in make_windows("Take 10 mg daily.", 50, count)] == ["Take 10 mg daily."]


def test_windows_split_at_sentences_and_respect_the_budget():
    body = "First sentence has five words. Second sentence also has six words. Third one is here now."
    windows = make_windows(body, 8, count)
    assert all(count(w.text) <= 8 for w in windows) and len(windows) == 3
    assert " ".join(w.text for w in windows) == body and not any(w.forced_split for w in windows)


def test_dose_is_never_separated_from_its_unit_in_a_forced_split():
    body = "word " * 6 + "500 mg twice daily with food and water"
    windows = make_windows(body.strip(), 7, count)
    assert any(w.forced_split for w in windows)
    assert any("500 mg" in w.text for w in windows) and not any(w.text.endswith("500") for w in windows)
    assert all(count(w.text) <= 7 for w in windows)


def test_table_rows_repeat_caption_and_header():
    body = "Table: Doses\n| Strength | Dose |\n| 10 mg | once daily |\n| 20 mg | twice daily |\n| 40 mg | at night |"
    windows = make_windows(body, 16, count)
    assert len(windows) > 1
    assert all(w.text.startswith("Table: Doses\n| Strength | Dose |") for w in windows)


def test_rerank_keeps_prefix_in_every_window_and_aggregates_by_max():
    long_body = "Unrelated text here. " * 6 + "Hepatotoxicity was reported."
    candidates = [
        {"chunk_id": "a", "prefix": "Drug A > WARNINGS", "text": "Nothing relevant."},
        {"chunk_id": "b", "prefix": "Drug B > BOXED WARNING", "text": long_body},
    ]
    scorer = KeywordScorer("hepatotoxicity")
    out = rerank("liver?", candidates, scorer, count, max_length=20, min_window_tokens=4)
    assert [r.chunk_id for r in out.ranked] == ["b", "a"]
    b = out.ranked[0]
    assert b.windows > 1 and b.score == 1 and b.original_rank == 2
    assert all(p.startswith("Drug B > BOXED WARNING\n") or p.startswith("Drug A > WARNINGS\n") for _, p in scorer.seen)
    assert all(count(q) + count(p) + 3 <= 20 for q, p in scorer.seen) and out.pairs == len(scorer.seen)


def test_ties_keep_the_original_fused_order():
    candidates = [{"chunk_id": c, "prefix": "P", "text": "same"} for c in ("x", "y", "z")]
    out = rerank("q", candidates, KeywordScorer("absent"), count, max_length=50)
    assert [r.chunk_id for r in out.ranked] == ["x", "y", "z"]


def test_too_long_question_and_prefix_is_an_error_not_a_silent_cut():
    with pytest.raises(ValueError):
        rerank("q " * 30, [{"chunk_id": "a", "prefix": "P " * 10, "text": "t"}], KeywordScorer("t"), count, 40)


def test_offline_metrics_count_evidence_items_once():
    spec = importlib.util.spec_from_file_location("eval_rerank", ROOT / "scripts/phase4/eval_rerank.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sets = [{"a1", "a2"}, {"b1"}]
    assert m.covered(["a1", "a2", "b1"], sets, 8) == [1, 0, 1]  # a2 is an overlap copy of the same item
    assert m.ap_at_k(["a1", "b1"], sets) == 1.0 and m.ap_at_k(["z", "a1", "b1"], sets) == pytest.approx(0.5833, 1e-3)
    assert m.ndcg_at_k(["a1", "b1"], sets) == 1.0 and m.recall_at_k(["z", "a1"], sets, 8) == 0.5
    assert m.first_rank(["z", "b1"], {"b1"}) == 2 and m.first_rank(["z"], {"b1"}) is None
