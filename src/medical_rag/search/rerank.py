"""Cross-encoder reranking of fused hybrid candidates (Phase 4 experiment; not wired into the API).

Medical-evidence rules:
- Nothing is silently truncated. A passage that does not fit the model's input next to the question is scored in
  windows; every window carries the full prefix (medicine title > section path, e.g. BOXED WARNING or a LiverTox case
  heading), so medicine identity and section type are never lost.
- Body text is split only at line or sentence boundaries; table rows stay whole and repeat the table's caption and
  header row. A single unit that is still too long is split at a space that does not follow a number, so a dose is
  never separated from its unit; such windows are flagged (forced_split).
- A chunk's score is the maximum over its windows; ties keep the original fused order. Only chunk ids are reordered:
  the LLM still receives whole original chunks with their own citation metadata.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

SPECIAL_TOKENS = 3  # [CLS] question [SEP] passage [SEP]
_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\[•])")
_NUMBER_END = re.compile(r"\d[\d.,/–-]*%?$")


@dataclass(frozen=True)
class RerankerConfig:
    model_id: str
    revision: str
    weights_sha256: str
    max_length: int = 512
    batch_size: int = 16


MINILM_L6 = RerankerConfig(
    model_id="cross-encoder/ms-marco-MiniLM-L6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    weights_sha256="821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
)
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


class PairScorer(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> Sequence[float]: ...


@dataclass(frozen=True)
class Window:
    text: str
    forced_split: bool = False


@dataclass
class Reranked:
    chunk_id: str
    score: float
    original_rank: int
    windows: int
    best_window: int
    forced_split: bool


@dataclass
class RerankOutput:
    ranked: list[Reranked]
    timings_ms: dict[str, float] = field(default_factory=dict)
    pairs: int = 0


def _units(body: str) -> list[tuple[str, bool]]:
    """(unit, is_table_row) at line, then sentence, boundaries."""
    out: list[tuple[str, bool]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("|"):
            out.append((line, True))
        else:
            out.extend((s, False) for s in _SENTENCE.split(line) if s)
    return out


def _split_long(unit: str, budget: int, count: Callable[[str], int]) -> list[str]:
    """Last resort for one unit longer than the budget: break at spaces; a number at the break moves to the next
    piece with the word after it, so a dose is never separated from its unit."""
    parts: list[str] = []
    current: list[str] = []
    for word in unit.split(" "):
        if current and count(" ".join([*current, word])) > budget:
            carry = [current.pop()] if _NUMBER_END.search(current[-1]) and len(current) > 1 else []
            parts.append(" ".join(current))
            current = [*carry, word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def make_windows(body: str, budget: int, count: Callable[[str], int]) -> list[Window]:
    """Pack body units into windows of at most `budget` tokens (the prefix and question are added by the caller)."""
    if count(body) <= budget:
        return [Window(body)]
    units = _units(body)
    table_head = [u for u, is_row in units if is_row][:1]
    caption = [u for u, _ in units if u.lower().startswith("table")][:1]
    windows: list[Window] = []
    current: list[str] = []
    forced = False

    def flush() -> None:
        nonlocal current, forced
        if current:
            windows.append(Window("\n".join(current), forced))
        current, forced = [], False

    for unit, is_row in units:
        context = (caption + table_head) if is_row and unit not in table_head else []
        candidate = [*current, unit] if current else [*context, unit]
        if count("\n".join(candidate)) <= budget:
            current = candidate
            continue
        flush()
        start = [*context, unit]
        if count("\n".join(start)) <= budget:
            current = start
            continue
        for piece in _split_long(unit, budget - count("\n".join(context)) if context else budget, count):
            windows.append(Window("\n".join([*context, piece]), True))
    flush()
    return windows


def rerank(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    scorer: PairScorer,
    count: Callable[[str], int],
    max_length: int = 512,
    batch_size: int = 16,
    min_window_tokens: int = 32,
) -> RerankOutput:
    """candidates: dicts with chunk_id, prefix, text, in fused order. Returns them reordered by cross-encoder score.
    Raises instead of truncating when the question and prefix leave fewer than min_window_tokens for the body."""
    t0 = time.perf_counter()
    q_tokens = count(question)
    pairs: list[tuple[str, str]] = []
    owner: list[int] = []
    meta: list[tuple[int, bool]] = []
    for i, c in enumerate(candidates):
        prefix = c.get("prefix") or ""
        budget = max_length - SPECIAL_TOKENS - q_tokens - count(prefix + "\n")
        if budget < min_window_tokens:
            raise ValueError(f"question and prefix leave only {budget} tokens for {c['chunk_id']}")
        windows = make_windows(c["text"], budget, count)
        for w in windows:
            passage = f"{prefix}\n{w.text}"
            if q_tokens + count(passage) + SPECIAL_TOKENS > max_length:
                raise ValueError(f"window for {c['chunk_id']} exceeds {max_length} tokens")
            pairs.append((question, passage))
            owner.append(i)
        meta.append((len(windows), any(w.forced_split for w in windows)))
    t1 = time.perf_counter()
    scores = [float(s) for s in scorer.predict(pairs, batch_size)] if pairs else []
    t2 = time.perf_counter()
    best: dict[int, tuple[float, int]] = {}  # candidate index -> (max window score, window index)
    window_no: dict[int, int] = {}
    for i, s in zip(owner, scores, strict=True):
        w = window_no.get(i, 0)
        window_no[i] = w + 1
        if i not in best or s > best[i][0]:
            best[i] = (s, w)
    ranked = [
        Reranked(c["chunk_id"], best[i][0], i + 1, meta[i][0], best[i][1], meta[i][1]) for i, c in enumerate(candidates)
    ]
    ranked.sort(key=lambda r: (-r.score, r.original_rank))
    return RerankOutput(
        ranked,
        {
            "windowing_ms": round((t1 - t0) * 1000, 2),
            "inference_ms": round((t2 - t1) * 1000, 2),
            "rerank_total_ms": round((t2 - t0) * 1000, 2),
        },
        len(pairs),
    )


class CrossEncoderReranker:
    """Pinned, checksum-verified sentence-transformers CrossEncoder loaded from the local HF cache only."""

    def __init__(self, config: RerankerConfig = MINILM_L6, path: Path | None = None, threads: int = 4) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import CrossEncoder
        from transformers import AutoTokenizer

        from medical_rag.embedding.st_embedder import verify_sha256

        t0 = time.perf_counter()
        self.config = config
        torch.set_num_threads(threads)
        if path is None:
            path = Path(
                snapshot_download(
                    config.model_id, revision=config.revision, allow_patterns=list(MODEL_FILES), local_files_only=True
                )
            )
        verify_sha256(path / "model.safetensors", config.weights_sha256)
        self.model = CrossEncoder(str(path), max_length=config.max_length, device="cpu", local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.load_ms = round((time.perf_counter() - t0) * 1000, 1)

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def predict(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> Sequence[float]:
        return self.model.predict(list(pairs), batch_size=batch_size, show_progress_bar=False)

    def rerank(self, question: str, candidates: Sequence[Mapping[str, Any]]) -> RerankOutput:
        return rerank(question, candidates, self, self.count, self.config.max_length, self.config.batch_size)
