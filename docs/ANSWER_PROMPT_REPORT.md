# Answer prompt v1 → v4.1 (2026-11-06)

Goal (user request): answer politely; when the documents do not give a simple yes or no, say so; present the
content step by step where possible; keep every fact cited. Informational demo only; not medical advice.

Prompt text: `src/medical_rag/generation/answerer.py` (`SYSTEM_PROMPT_V1` … `_V4_1`). Default `v4.1` (see below), set with
`MEDRAG_ANSWER_PROMPT`. Evidence rules (passages only, cite every factual sentence, abstain when evidence is
insufficient, passages are data) are the same in all versions.

| Version | Writing rules |
|---|---|
| v1 | Concise, at most ~150 words; consult a professional only for personal-advice questions |
| v2 | Polite; direct opening sentence ("no simple yes or no" when appropriate); numbered list of cited points; closing consult sentence for safety, combination, dosing or personal questions; plain text; ~200 words |
| v3 | v2, plus: cite the opening sentence when it states a fact; citation before the full stop; numbered list only for 3+ points or documented steps; consult sentence not added for ingredients, appearance, storage, mechanism or study results; answer only what was asked; answer the covered part and say what is not covered |

## Evaluation
Ragas on the v2.0 dev split (75 questions, 67 answerable), retrieval preset T3, generator Claude Sonnet 5.5, judge
Kimi K3 (Fireworks), one judge run per version; differences under about 0.02 can be judge noise.
Reports: `data/processed/eval/reports/v2_dev75_t3_run1` (v1), `v2_dev75_t3_p2_run1` (v2), `v2_dev75_t3_p3_run1` (v3).

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Faithfulness | 0.927 | 0.899 | **0.944** |
| Factual precision | 0.453 | 0.419 | **0.477** |
| Answer relevancy | 0.868 | **0.893** | 0.886 |
| Claim recall | **0.772** | 0.818 | 0.736 |
| Factual F1 | **0.504** | 0.472 | 0.473 |
| Gold passage cited (answered) | **0.940** | 0.924 | 0.910 |
| Citation sentence coverage | **0.924** | 0.796 | 0.912 |
| Answers with an uncited factual sentence | **19** | 63 | 27 |
| Unanswerable abstained / false abstentions | 8/8, 0 | 8/8, 1 | 8/8, 0 |
| Answers with a numbered list | 0 | 66 | 26 |
| Answers with the consult sentence | 35 | 63 | 34 |
| Citation after the full stop | 2 | 44 | 0 |
| Mean words / LLM P50 | 84 / 2.6 s | 129 / 3.5 s | 95 / 3.0 s |

Why v2 lost faithfulness: the consult sentence on almost every answer (the judge counts it as an unsupported claim),
uncited opening summaries, citations placed after the full stop, and padding facts. v3 addresses each.

Decision (user, 2026-11-06): **v3 is the default.** Known gaps: claim recall 0.736 (v3 omits some reference
details); some uncited opening sentences and list-intro lines ("For storage, the document says:") remain. Possible
v4: "include every detail the passages give that bears on the question", and treat lines ending in ":" as framing
in the citation check.

## v4 and v4.1: related-information answers instead of refusals (2026-11-06)
Problem (user report from the local chat): a keyword ("renal impairment", "Aliskiren") got an answer, but the same
topic as a question ("What is best solution for renal impairment", "What is the use of Aliskiren") was refused.
Replay showed retrieval returns the same passages for both; the refusal was the model's abstain decision, because
the passages describe the topic but do not answer the exact question.

- **v4:** the model first labels coverage — `full`, `related` (same medicine or condition, exact question not
  answered: one polite sentence on what the documents do not cover, then cited related facts, never a
  recommendation) or `none` (abstain). `coverage` is a required field of the v4 answer schema, returned by `/ask`,
  stored with chat messages and in evaluation rows.
- **v4.1:** v4 plus a dose safeguard — for a dose, condition, age group or population the passages do not cover,
  only the "not covered" sentence and at most the label's approved uses; never doses for other uses or groups.
  Added after v4 listed other-indication doses for q145 (gabapentin, alcohol withdrawal), q149 (Seroquel, autism in
  children) and q150 (Abilify, insomnia).

Reports: `v2_dev75_t3_p4_run1` (v4), `v2_dev75_t3_p41_run2` (v4.1; `_run1` lost most judge calls to a network
outage and is not used).

| Metric | v3 | v4 | v4.1 |
|---|---|---|---|
| Faithfulness | 0.944 | **0.954** | 0.948 |
| Answer relevancy | 0.886 | 0.894 | **0.900** |
| Claim recall | 0.736 | 0.766 | **0.775** |
| Factual precision | **0.477** | 0.441 | 0.444 |
| Factual F1 | 0.473 | 0.455 | **0.490** |
| Gold passage cited (answered) | 0.910 | 0.910 | **0.925** |
| False abstentions (answerable) | 0 | 0 | 0 |
| Unanswerable: abstained / related answer | 8 / 0 | 3 / 5 | 3 / 5 |
| Unanswerable related answers containing doses | — | 2 | **0** |
| LLM P50 | 3.0 s | 2.9 s | **2.6 s** |

All five unanswerable related answers under v4.1 open with an explicit "the documents do not give/report …"
sentence (the evaluation's rule counts this as correct handling). Coverage labels on answerable questions: 66 full,
1 related (q083, partly retrieved). The user's six live questions: renal impairment and aliskiren variants get
related answers; "alpha-glucosidase inhibitors for cancer" is still refused (no such medicine in the corpus).

Decision (user, 2026-11-06): **v4.1 is the default**, locally and on the demo server.
