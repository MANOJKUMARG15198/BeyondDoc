"""Settings-driven retrieval plan selection for the API and chat (Retrieval v1.2). Off unless MEDRAG_RETRIEVAL_PLAN
is set; the alias table is loaded once per path."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from medical_rag.core.config import Settings
from medical_rag.query.medicine import AliasTable
from medical_rag.search.planner import RetrievalPlan, plan_question

ALIAS_DIR = Path("data/processed/medicine")


@lru_cache(maxsize=4)
def _table(path: str) -> AliasTable:
    return AliasTable.load(Path(path))


def alias_table_path(settings: Settings) -> str:
    if settings.alias_table_path:
        return settings.alias_table_path
    found = sorted(ALIAS_DIR.glob("aliases_*.json"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no alias table in {ALIAS_DIR}; run scripts/phase5/build_medicine_aliases.py")
    return str(found[-1])


def plan_for(question: str, settings: Settings, mode: str = "hybrid") -> RetrievalPlan | None:
    """The configured plan for a hybrid query, or None (plain hybrid search) when no preset is enabled."""
    if settings.retrieval_plan == "none" or mode != "hybrid":
        return None
    return plan_question(question, _table(alias_table_path(settings)))
