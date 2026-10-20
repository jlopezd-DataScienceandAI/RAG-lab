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

The step-by-step plan lives in a single place: [v0.1-guide.md](v0.1-guide.md). Follow its
steps in order and do not move on until the check of the current step passes. This file
only covers the git workflow and how to run experiments, so the plan is not duplicated here.

## Running experiments

- Change one variable at a time (chunk size, k, embedding model, prompt version).
- Every run writes its own JSON file with the full configuration and the metrics.
- Never overwrite earlier results; they are the evidence the README links to.

## Daily commands

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```
