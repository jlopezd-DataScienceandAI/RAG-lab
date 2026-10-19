---
description: Review my uncommitted changes before I commit
---

Review my uncommitted changes as a senior engineer would, before I commit.

1. Run `git status` and `git diff` (and `git diff --staged`) to see what changed.
2. Run `uv run ruff check .` and `uv run pytest -m "not slow"` and report the results.
3. Check, in this order:
   - **Bugs**: wrong logic, edge cases (empty input, k larger than the store, missing files),
     off-by-one errors, ranks or indices out of order.
   - **Safety**: `.env`, API keys, `index/` or large files about to be committed; the API key in
     anything written to `results/`.
   - **Project rules** in CLAUDE.md (VectorStore interface, `row_idx` alignment, prompt
     versioning, results never overwritten).
   - **Tests**: is the new behaviour tested? Suggest the missing test cases.
   - **Readability**: names, type hints, docstrings, function size.
4. Suggest a commit message with the right prefix (`feat:`, `fix:`, `test:`, `eval:`, `docs:`).

List findings from most to least important, each with the file and line and a one-line fix.
Explain the reason behind each one so I learn from it. Do not edit files and do not commit.
Reply in Spanish.
