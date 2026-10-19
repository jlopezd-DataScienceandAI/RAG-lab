## What changes

<!-- One or two sentences. -->

## Why

<!-- Which limitation from docs/limitations.md motivates this, and what hypothesis is being tested? -->

## Results

Evaluated on `data/eval/questions.jsonl` (same dataset for all versions).

| Metric | Before | After |
|---|---|---|
| Hit@5 | | |
| MRR | | |
| Faithfulness | | |
| Latency (p50) | | |

Results file: `results/<version>/<file>.json`

## New limitations found

<!-- Anything this change did not fix, or introduced. It feeds the next version. -->

## Checklist

- [ ] Tests pass (`pytest`) and `ruff check .` is clean
- [ ] Results saved in `results/`
- [ ] `CHANGELOG.md` updated
- [ ] `docs/limitations.md` updated
- [ ] `docs/decisions.md` updated if a design decision changed
- [ ] README results table updated
