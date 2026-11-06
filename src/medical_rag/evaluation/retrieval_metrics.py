"""Retrieval metrics over ranked chunk ids: hit@k, recall@k, MRR@k, nDCG@k (binary relevance)."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(c in relevant for c in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, c in enumerate(ranked[:k], start=1):
        if c in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, c in enumerate(ranked[:k], start=1) if c in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


METRICS = {"hit": hit_at_k, "recall": recall_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k}


def score_run(
    rankings: Iterable[tuple[Sequence[str], set[str]]], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    """Mean of each metric at each k over (ranked ids, relevant ids) pairs; queries without gold are skipped."""
    rows = [(r, g) for r, g in rankings if g]
    out: dict[str, float] = {"queries": float(len(rows))}
    for name, fn in METRICS.items():
        for k in ks:
            out[f"{name}@{k}"] = round(statistics.fmean(fn(r, g, k) for r, g in rows), 4) if rows else 0.0
    return out
