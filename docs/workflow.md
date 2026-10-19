# Development workflow

## Git: one branch per version, a tag per release

`main` always holds the latest working code. Versions are tags, not folders.

```bash
# once, at the start
git init -b main
git add . && git commit -m "chore: project skeleton"
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin main

# for each version
git switch -c v0.1-baseline
# ... small commits (see below) ...
git push -u origin v0.1-baseline
# open a pull request to main, fill in the template, merge it

git switch main && git pull
git tag -a v0.1 -m "Baseline RAG: numpy + SQLite"
git push origin v0.1
# then create a GitHub Release from the tag (.github/RELEASE_TEMPLATE.md)
```

Compare two versions at `https://github.com/<user>/<repo>/compare/v0.1...v0.3`.

### Commit messages

`feat:` new behaviour, `fix:` bug fix, `test:` tests, `eval:` evaluation code or results,
`docs:` documentation, `refactor:` no behaviour change, `chore:` tooling.
Examples: `feat: add token-based chunking`, `eval: record v0.1 baseline results`.

## Build order for v0.1

The evaluation dataset comes before the pipeline. Commit after each step.

| Step | What to do | Done when |
|---|---|---|
| 1 | Choose the corpus (10-30 English documents) into `data/raw/` | Documents are in place |
| 2 | Write 20-30 questions in `data/eval/questions.jsonl` (3-5 unanswerable) | `pytest tests/test_smoke.py` passes with your data |
| 3 | `load_document` in `ingest/loaders.py` | Text and page numbers extracted from every file |
| 4 | `chunk_pages` in `ingest/chunking.py` | `tests/test_chunking.py` passes (remove xfail); chunks read well by eye |
| 5 | `Embedder` in `embeddings.py` | Shape and normalisation checked in a test |
| 6 | `NumpyStore` in `store/numpy_store.py` | Searching a chunk's own text returns that chunk first; save/load round-trips |
| 7 | `rag ingest` in `cli.py`, with `doc_hash` to skip unchanged files | Re-running ingest does nothing |
| 8 | `retrieve` and the metrics | `tests/test_metrics.py` passes; first Hit@k and MRR on your dataset |
| 9 | `generation.py` with citations | Answers cite chunk ids; unanswerable questions get "I don't know" |
| 10 | `judge.py` and `run_eval.py` | One JSON per run in `results/v0.1/` |
| 11 | Failure analysis | Every failing question classified: chunking, retrieval or prompt |
| 12 | Measure the limitations in `docs/limitations.md` | *Measured* column filled |
| 13 | README table, CHANGELOG, PR, tag, Release | v0.1 published |

Do not touch Claude until step 8 shows that retrieval works: if the right chunk is not
retrieved, the generation cannot be right.

## Running experiments

- Change one variable at a time (chunk size, k, embedding model, prompt version).
- Every run writes its own JSON file with the full configuration and the metrics.
- Never overwrite earlier results; they are the evidence the README links to.

## Daily commands

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```
