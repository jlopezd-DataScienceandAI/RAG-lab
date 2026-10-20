# Changelog

All notable changes are recorded here, newest first. Each version states what changed,
why, and which limitation (see `docs/limitations.md`) motivated it.

## [Unreleased]

### Added
- Project skeleton: package layout, configuration, data models, vector store interface,
  prompt template, evaluation dataset format, tests and PR/release templates.
- Step-by-step v0.1 guide (`docs/v0.1-guide.md`), `CLAUDE.md` and the `/hint` and `/review`
  commands for working in learning mode.
- Initial corpus in `data/raw/`: AI Act, GDPR, default-definition guidelines and PD/LGD
  estimation guidelines (source list pending, see README).
- First 17 hand-written evaluation questions in `data/eval/questions.jsonl` (14 answerable,
  3 unanswerable). To be extended once the corpus is closed.
- Document loading for PDF, Markdown and text (`load_document`, `file_sha256`), with tests.
- Paragraph-aware token chunking with overlap (`chunk_pages`), with tests.
- `uv.lock` lockfile.
- Design decisions D-008 to D-014 in `docs/decisions.md`.

### Changed
- Project name unified as `rag-lab` (README, `pyproject.toml`, lockfile).
- `docs/workflow.md` no longer repeats the build order: `docs/v0.1-guide.md` is the single
  source.
- `.gitignore` now ignores `.claude/settings.local.json`.

## [0.1.0] - TBD

Baseline RAG pipeline.

### Added
- Document loading (PDF, Markdown), chunking by tokens with overlap
- Sentence-aware chunking as a second strategy, compared with paragraph chunking on the
  same dataset (only the chunking changes)
- Local embeddings with sentence-transformers
- Exact cosine search over a numpy matrix, metadata in SQLite
- Answer generation with Claude and chunk-id citations
- Evaluation: Hit@k, MRR, judged faithfulness/correctness, citation check, abstention

### Results
| Hit@5 | MRR | Latency | Dataset size |
|---|---|---|---|
| TBD | TBD | TBD | TBD |
