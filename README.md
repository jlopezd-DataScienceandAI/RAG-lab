# rag-project

A small Retrieval-Augmented Generation (RAG) pipeline built from scratch: no LangChain, no
LlamaIndex, no vector database. Every piece (chunking, retrieval, prompting, evaluation) is
written by hand so that its behaviour and limits are understood, measured, and then improved
one version at a time.

> Status: **v0.1 in progress.** See [CHANGELOG.md](CHANGELOG.md).

## Versions and results

Every version is evaluated on the same dataset (`data/eval/questions.jsonl`).

| Version | Main change | Hit@5 | MRR | Latency | Details |
|---|---|---|---|---|---|
| v0.1 | Baseline: numpy + SQLite, dense retrieval, cited answers, paragraph chunking | TBD | TBD | TBD | TBD |
| v0.1 | Same baseline with sentence-aware chunking (only change) | TBD | TBD | TBD | TBD |

Each limitation found in one version motivates the next one: see
[docs/limitations.md](docs/limitations.md).

## How it works (v0.1)

```
Indexing (rag ingest)
  documents -> chunking -> embeddings -> index/ (SQLite + vectors.npy)

Querying (rag ask)
  question -> embed -> top-k chunks -> prompt -> Claude -> answer with citations

Evaluation (rag eval)
  questions.jsonl -> retrieval metrics (Hit@k, MRR) + judged answers -> results/
```

## Quickstart

```bash
uv sync --extra dev            # or: python -m venv .venv && pip install -e ".[dev]"
# create a .env file in the repo root containing: ANTHROPIC_API_KEY=<your key>
# put your documents in data/raw/
uv run rag ingest
uv run rag ask "What is ...?"
uv run rag eval
```

## Project layout

```
data/raw/            source documents
data/eval/           evaluation questions
index/               generated index (git-ignored)
prompts/             versioned prompt templates
src/rag/             the pipeline (ingest, embeddings, store, retrieval, generation, evaluation)
tests/               unit tests
results/<version>/   evaluation output, kept as historical evidence
docs/                design decisions, limitations, workflow
```

## Roadmap

- **v0.1** baseline RAG with its own evaluation, plus a measured comparison of paragraph vs.
  sentence-aware chunking
- **v0.2** API (FastAPI) and a small demo UI
- **v0.3** hybrid search (BM25 + vectors), reranking, vector-store comparison, ragas
- **v0.4** agent with tools
- **v0.5** Docker, CI, tracing of cost and latency

## Documentation

- [Design decisions](docs/decisions.md)
- [Limitations and the experiments that measure them](docs/limitations.md)
- [Development workflow](docs/workflow.md)
