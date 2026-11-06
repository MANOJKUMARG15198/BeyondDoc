"""Reference-claim recall with a fake judge (no API calls)."""

import asyncio

import pytest
from ragas.metrics.collections.factual_correctness.util import ClaimDecompositionOutput, NLIStatementOutput

from medical_rag.evaluation.claim_recall import ClaimRecall


class FakeJudge:
    def __init__(self, claims, verdicts):
        self.claims, self.verdicts, self.prompts = claims, verdicts, []

    async def agenerate(self, prompt, response_model):
        self.prompts.append(prompt)
        if response_model is ClaimDecompositionOutput:
            return ClaimDecompositionOutput(claims=self.claims)
        assert response_model is NLIStatementOutput
        return NLIStatementOutput(
            statements=[  # strict=False on purpose: the mismatch test returns fewer verdicts than claims
                {"statement": c, "reason": "r", "verdict": v} for c, v in zip(self.claims, self.verdicts, strict=False)
            ]
        )


def run(judge, response="Answer [S1].", reference="Reference."):
    return asyncio.run(ClaimRecall(judge).ascore(response=response, reference=reference))


def test_recall_is_share_of_reference_claims_found_in_answer():
    judge = FakeJudge(["a", "b", "c", "d"], [1, 1, 1, 0])
    result = run(judge, response="THE ANSWER", reference="THE REFERENCE")
    assert result.value == 0.75 and result.verdicts == [1, 1, 1, 0] and not result.verdict_count_mismatch
    # reference is decomposed; the claims are checked against the answer
    assert "THE REFERENCE" in judge.prompts[0] and "THE ANSWER" in judge.prompts[1]


def test_no_reference_claims_gives_none():
    assert run(FakeJudge([], [])).value is None


def test_verdict_count_mismatch_is_flagged():
    result = run(FakeJudge(["a", "b"], [1]))  # two claims, the judge returns one verdict
    assert result.verdict_count_mismatch and result.value == 1.0


def test_empty_inputs_rejected():
    with pytest.raises(ValueError):
        run(FakeJudge(["a"], [1]), response=" ")
