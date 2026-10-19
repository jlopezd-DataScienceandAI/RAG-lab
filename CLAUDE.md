# CLAUDE.md

Context for Claude Code when working in this repository.

## About me and how to help

- I am a student building this project to learn AI engineering and to show it in internship
  applications. **Reply to me in Spanish.** Code, comments, commit messages and docs stay in
  English.
- **Learning mode.** Functions marked `TODO(v0.1)` are mine to implement. Unless I explicitly
  say "write the implementation", do not write them for me: explain the concept, give hints,
  point to the relevant line, or show a tiny standalone example that is not the solution.
- You **may** write tests, fix tooling and configuration, explain errors, and review my code.
- When you review, be direct: list real bugs first, then risks, then style. Say what is
  already good, briefly.
- Explain the *why* of each suggestion. I want to be able to defend every line in an interview.
- Never commit, push, create branches, tags or releases yourself. Suggest the git commands and I
  run them.

## Project

`rag-lab` is a Retrieval-Augmented Generation pipeline built from scratch (no LangChain, no
LlamaIndex, no vector database) and improved one measured version at a time. The current
version is **v0.1**. The step-by-step plan is in `docs/v0.1-guide.md`; check which step I am on
before suggesting work from later steps.

Pipeline: `ingest` (load -> chunk -> embed -> save index) / `ask` (embed query -> top-k ->
prompt -> Claude -> cited answer) / `eval` (retrieval metrics + LLM judge -> `results/`).

## Layout

- `src/rag/config.py`: settings from `.env` / env vars (prefix `RAG_`)
- `src/rag/models.py`: `Chunk`, `RetrievalResult`, `EvalCase`
- `src/rag/ingest/`: `loaders.py` (PDF/Markdown), `chunking.py`
- `src/rag/embeddings.py`: `Embedder` (sentence-transformers, `BAAI/bge-small-en-v1.5`)
- `src/rag/store/`: `base.py` (`VectorStore` interface), `numpy_store.py` (numpy + SQLite)
- `src/rag/retrieval.py`, `src/rag/generation.py`
- `src/rag/evaluation/`: `metrics.py`, `judge.py`, `run_eval.py`
- `src/rag/cli.py`: Typer CLI (`rag ingest | ask | eval`)
- `prompts/`: versioned prompt templates; never edit an old version, add `_v2`
- `data/raw/`: corpus; `data/eval/questions.jsonl`: hand-written evaluation set
- `index/`: generated, git-ignored; `results/<version>/`: evaluation JSON, committed
- `docs/`: `decisions.md`, `limitations.md`, `workflow.md`, `v0.1-guide.md`

## Commands

All commands run from the repository root, on Windows PowerShell.

```powershell
uv run pytest                 # all tests
uv run pytest -m "not slow"   # skip tests that download models
uv run ruff format .
uv run ruff check .
uv run rag ingest
uv run rag ask "question"
uv run rag eval [--retrieval-only]
```

## Rules

- Never read, print or modify `.env`, and never put the API key in code, logs or results.
- Do not add dependencies without asking; justify any new one against `docs/decisions.md`.
- Everything outside `store/` talks to the `VectorStore` interface, never to numpy directly.
- Keep `row_idx` in SQLite aligned with the row order of `vectors.npy`.
- Evaluation always uses `data/eval/questions.jsonl`. Change one variable per experiment and
  never overwrite files in `results/`.
- Do not modify `data/eval/questions.jsonl` unless I ask: it is the ground truth.
- Python 3.11+, type hints everywhere, ruff line length 100.
