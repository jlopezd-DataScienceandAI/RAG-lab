# Changelog

All notable changes are recorded here, newest first. Each version states what changed,
why, and which limitation (see `docs/limitations.md`) motivated it.

## [Unreleased]

### Added
- Project skeleton: package layout, configuration, data models, vector store interface,
  prompt template, evaluation dataset format, tests and PR/release templates.

## [0.1.0] - TBD

Baseline RAG pipeline.

### Added
- Document loading (PDF, Markdown), chunking by tokens with overlap
- Local embeddings with sentence-transformers
- Exact cosine search over a numpy matrix, metadata in SQLite
- Answer generation with Claude and chunk-id citations
- Evaluation: Hit@k, MRR, judged faithfulness/correctness, citation check, abstention

### Results
| Hit@5 | MRR | Latency | Dataset size |
|---|---|---|---|
| TBD | TBD | TBD | TBD |
