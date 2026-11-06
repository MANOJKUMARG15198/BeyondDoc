# First Claude Code Task - Phase 0

Paste this prompt into Claude Code from the repository root after copying the planning files into the repo.

---

Read `CLAUDE.md`, `docs/EXECUTION_CHECKLIST.md`, and `docs/DECISION_LOG.md` first.

The project is a medical-domain, public-document RAG platform with Python/FastAPI, OpenSearch, dense+sparse hybrid retrieval, Ragas evaluation, and a reproducible GitHub client demo. All unresolved model, chunking, indexing, fusion, and LLM selections must remain marked as pending; do not make those architecture decisions on the project owner's behalf.

**Task: Implement Phase 0 only (repository foundation).**

1. Start in plan mode. Show a concise directory layout, dependency choices and commands before editing.
2. Scaffold a maintainable Python package (e.g. `src/medical_rag/`) with `api`, `core`, `ingestion`, `chunking`, `embeddings`, `search`, `generation`, `evaluation`, and `observability` boundaries. Empty modules or documented interfaces are acceptable for future phases; avoid artificial implementation.
3. Implement a minimal FastAPI `/health` endpoint and tests for it. Add typed settings with env-based configuration and `.env.example`.
4. Create a pinned dependency setup, test/lint/format commands, a local-only Docker Compose OpenSearch service with health/readiness checks, GitHub Actions CI, a sensible `.gitignore`, and a quickstart README.
5. Document architecture modules, local development assumptions and how to run verification. Do not pull large medical datasets, download models, or implement retrieval/generation yet.
6. Run the available checks. Return: changed files; exact commands and outcomes; known issues; phase-0 checklist changes; and recommended next single task. Do not claim success if something was not executed.

**Security and quality:** Use no patient-identifiable records. Never commit secrets. Use explicit local-dev-only settings where needed. Pin/test compatible OpenSearch and Python versions, but record the versions as implementation choices subject to review. Do not install unrequested cloud services or publish/deploy anything.

---
