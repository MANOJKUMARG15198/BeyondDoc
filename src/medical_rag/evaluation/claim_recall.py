"""Reference-claim recall: the share of the reference answer's claims that the generated answer states.

Why not Ragas FactualCorrectness(mode="recall"): in ragas 0.4.3 its true positives come from the *answer's* claims
checked against the reference (the precision direction) and only its false negatives come from the reference's
claims, so recall = tp_answer / (tp_answer + fn_reference). An answer that states every reference fact but whose own
claims the judge does not match to the one-sentence reference scores 0. This metric uses one direction only:
decompose the reference into claims, then check each claim against the answer. It reuses Ragas's own claim
decomposition and NLI prompts so the judge instructions are unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ragas.metrics.collections.factual_correctness.util import (
    ClaimDecompositionInput,
    ClaimDecompositionOutput,
    ClaimDecompositionPrompt,
    NLIStatementInput,
    NLIStatementOutput,
    NLIStatementPrompt,
)


@dataclass
class ClaimRecallResult:
    value: float | None
    claims: list[str] = field(default_factory=list)
    verdicts: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    verdict_count_mismatch: bool = False  # judge returned a different number of verdicts than claims

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClaimRecall:
    name = "claim_recall"

    def __init__(
        self,
        llm: Any,  # an InstructorBaseRagasLLM, e.g. ClaudeJudge
        atomicity: Literal["low", "high"] = "low",  # same defaults as Ragas FactualCorrectness
        coverage: Literal["low", "high"] = "low",
    ) -> None:
        self.llm, self.atomicity, self.coverage = llm, atomicity, coverage
        self.decompose_prompt = ClaimDecompositionPrompt()
        self.nli_prompt = NLIStatementPrompt()

    async def ascore(self, response: str, reference: str) -> ClaimRecallResult:
        if not response.strip() or not reference.strip():
            raise ValueError("response and reference are required")
        decomposition = ClaimDecompositionInput(response=reference, atomicity=self.atomicity, coverage=self.coverage)
        claims = (
            await self.llm.agenerate(self.decompose_prompt.to_string(decomposition), ClaimDecompositionOutput)
        ).claims
        if not claims:
            return ClaimRecallResult(None)
        nli = NLIStatementInput(context=response, statements=claims)
        verdicts = (await self.llm.agenerate(self.nli_prompt.to_string(nli), NLIStatementOutput)).statements
        values = [int(bool(v.verdict)) for v in verdicts]
        return ClaimRecallResult(
            value=round(sum(values) / len(values), 4) if values else None,
            claims=list(claims),
            verdicts=values,
            reasons=[v.reason for v in verdicts],
            verdict_count_mismatch=len(values) != len(claims),
        )
