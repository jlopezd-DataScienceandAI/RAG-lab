# Evaluation methodology

This is the single source for **how rag-lab is evaluated**: what is measured, on which
questions, how each number is computed, how a run is executed and recorded, and how results are
compared between versions. Other documents only link here:

- The format of the questions is in [`data/eval/README.md`](../data/eval/README.md).
- Why a choice was made is in [`decisions.md`](decisions.md).
- Which weaknesses are expected, and which experiment measures each, is in
  [`limitations.md`](limitations.md).

Status: **draft for review.** Anything marked *(open)* is a proposal that has not been decided.

---

## 1. Principles

1. **One fixed dataset.** Every version is evaluated on `data/eval/questions.jsonl`. If the
   dataset changes, every earlier version is re-run and the change is noted in `CHANGELOG.md`.
2. **One variable per experiment.** Between two runs that are compared, exactly one thing
   changes (chunking, prompt, model, `k`, ...). Everything else stays identical.
3. **Retrieval is measured on its own.** The retrieval step runs without any LLM, because if
   the right chunk is not retrieved, no prompt can fix the answer. Generation is measured
   afterwards, on top of it.
4. **Results are immutable.** Every run writes a new timestamped JSON file under
   `results/<version>/`. Existing files are never overwritten or edited. A bad run is
   explained in the notes, not deleted.
5. **Numbers justify changes.** A change is accepted because a measured number moved, and the
   before/after numbers are written in `limitations.md` and `decisions.md`.
6. **Questions are written before looking at results.** The dataset is written from the
   documents, never after seeing what the retriever returns, so it is not biased towards what
   the system already does well (D-012).
7. **No secrets in results.** The API key is excluded from the saved configuration.

---

## 2. What is evaluated

The pipeline has three stages, and each is scored separately so a failure can be located:

| Stage | Question it answers | Needs an LLM? |
|---|---|---|
| Retrieval | Did the right chunk come back, and how high? | No |
| Generation | Is the answer correct, supported by the context, and properly cited? | Yes (generator and judge) |
| System | How long does it take, end to end? | Yes |

Questions are split by whether the corpus can answer them (`answerable`) and by type (`tags`).
The same run reports metrics overall and per tag.

---

## 3. The dataset in one paragraph

20-30 hand-written questions over the 11 PDFs in `data/raw/`: 11-15 `factual`, 4-6
`exact_term`, 2-4 `multi_hop` and 3-5 `unanswerable`. Each answerable question has a
`gold_doc`, a literal `gold_evidence` phrase (5-15 words) and a short `expected_answer`.
Unanswerable questions have those three fields set to `null` and test abstention. The exact
format, rules and examples are in [`data/eval/README.md`](../data/eval/README.md).

Why the tags matter: `exact_term` questions are where dense retrieval is expected to fail (L1),
`multi_hop` questions need two chunks, and `unanswerable` questions are the only way to measure
hallucination and abstention.

---

## 4. Definitions

### 4.1 A relevant chunk

A retrieved chunk is **relevant** for a question when:

1. The question has a `gold_evidence`: the normalised evidence is contained in the normalised
   chunk text. Normalisation lowercases and collapses all whitespace
   (`" ".join(text.lower().split())`), because PDF extraction inserts line breaks inside
   sentences.
2. Otherwise: `chunk.doc == gold_doc`. This is looser (a whole document counts), so questions
   should carry `gold_evidence` whenever they have a single literal source.

Consequence: a chunking change can move the evidence across a chunk boundary and make a
correct retrieval look like a miss. That is why `gold_evidence` is short, and why a failed
question is always read by hand before being blamed on the retriever (see section 9).

Words split by the PDF extractor (`Int\nroductory`) are **not** repaired by normalisation, so
the evidence must be copied from the text as `pypdf` extracts it (limitation L7).

### 4.2 Retrieval metrics (answerable questions only)

Computed over the top-`k` results (`k = settings.top_k`, 5 by default):

| Metric | Definition |
|---|---|
| `hit@1` | 1 if the first result is relevant, else 0 |
| `hit@5` | 1 if any of the first 5 results is relevant, else 0 |
| `rr` (reciprocal rank) | `1 / rank` of the first relevant result, 0 if none |

Aggregates are means over the answerable questions: `hit@1`, `hit@5` and `mrr` (mean of `rr`).
Unanswerable questions are skipped here because they have no correct chunk.

Reading them together:

- `hit@5` low: the right chunk is not retrieved at all. Look at chunking, loading and the
  embedding model.
- `hit@5` high but `mrr` low: it is retrieved but ranked badly (L2, motivates reranking).
- Reference point: for a small clean corpus, `hit@5` below about 0.7 usually means a bug in
  loading or chunking, not a weak model.

### 4.3 Generation metrics

The generator answers using only the retrieved chunks and cites them as `[chunk_id]`. The
judge is a second LLM call with a fixed rubric (`prompts/judge_v1.txt`), which returns JSON:

| Verdict field | Meaning |
|---|---|
| `faithful` | Every factual claim is supported by the retrieved context. An abstention counts as faithful |
| `correct` | The answer conveys the `expected_answer`. For an unanswerable question, correct means the system abstained |
| `abstained` | The answer says it does not know |

The judge gets the expected answer as text. For unanswerable questions `expected_answer` is
`null`, so the runner passes the literal string `"none"` and the rubric defines correct as
abstaining in that case.

Citations are checked **programmatically**, not by the judge: the answer is scanned for
`[id]` markers and the check passes if every cited id is among the retrieved chunks. This
catches invented ids; it does not prove the cited chunk supports the claim (that is what
`faithful` is for).

### 4.4 Aggregate metrics

| Metric | Over | Definition |
|---|---|---|
| `hit@1`, `hit@5`, `mrr` | answerable | See 4.2 |
| `faithfulness` | all | Share of answers with `faithful = true` |
| `correctness` | answerable | Share of answers with `correct = true` |
| `citation_validity` | answerable, not abstained | Share whose cited ids are all among the retrieved chunks |
| `abstention_accuracy` | unanswerable | Share where `abstained = true` |
| `false_abstention` | answerable | Share where the system abstained without needing to |
| `latency_ms_p50`, `latency_ms_p95` | all | End-to-end latency percentiles |
| `judge_parse_errors` | all | Number of verdicts that could not be parsed |

Every share is reported with its **count** (for example `15/20`), not only the percentage.
With about 20 answerable questions, one question is worth 5 points; a difference of one or two
questions between runs is noise, not a result.

*(open)* `citation_validity` when the answer abstains and cites nothing: the proposal is to
exclude abstentions from the denominator, as written above.

### 4.5 Latency

Measured with `time.perf_counter()` per question, in milliseconds, for retrieval and for
generation separately, plus the total. Reported as p50 and p95
(`float(np.percentile(values, q))`). The first question includes loading the embedding model,
so the model is warmed up with one throwaway query before the timed loop, otherwise p95 is
dominated by the load time.

---

## 5. Models and configuration

| Role | Model | Configured in | Why |
|---|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | `embedding_model` | Local, free, 384 dimensions (D-002) |
| Generator | `claude-haiku-5-5` | `generation_model` | Cheap while iterating (D-005) |
| Judge | `claude-sonnet-5-5` | `judge_model` | Stronger judge for the numbers that get published |

Fixed during a comparison unless it is the variable under test: `chunk_tokens = 500`,
`chunk_overlap = 75`, `top_k = 5`, the answer prompt (`prompts/answer_v1.txt`), the judge
prompt and `max_answer_tokens = 600`. The `temperature` parameter is not sent to the API
(recent models are retiring it), so sampling uses the API default for generator and judge.

Two kinds of run:

- **Iteration runs:** generator Haiku, judge Sonnet. Used while developing.
- **Final run:** the one that goes into the README. Same generation configuration as the last
  iteration run; it is the only one quoted in the version table.

*(open)* Whether the final run should use Sonnet as generator too. Changing the generator
between iteration and final run would break "one variable at a time", so the proposal is to keep
Haiku as generator for every v0.1 run and use Sonnet only as judge.

A run's configuration is saved inside its own JSON file (see section 7), so a result is never
separated from the settings that produced it.

---

## 6. Procedure

### 6.1 Before the first run (once per dataset)

1. Close the corpus in `data/raw/` (no more documents added or replaced).
2. Run `uv run rag ingest` so the index matches the corpus. The command is a no-op if nothing
   changed.
3. Write `questions.jsonl` by reading the documents, following the format README.
4. Run the dataset tests:

   ```powershell
   uv run pytest tests/test_smoke.py             # format, tags, doc exists, counts
   uv run pytest tests/test_dataset_evidence.py  # every gold_evidence is in its document (slow)
   ```

5. Tag the dataset state in the changelog (date and number of questions per tag).

### 6.2 Every evaluation run

1. **Clean tree.** `git status` shows no uncommitted change that affects the pipeline, apart
   from the one being tested. The result file records the commit hash so it can be traced.
2. **Index up to date.** `uv run rag ingest` prints "Index is up to date" (or rebuilds it).
   Never evaluate against a stale index.
3. **Run retrieval only** (fast, free, deterministic):

   ```powershell
   uv run rag eval --retrieval-only
   ```

   Inspect it before spending any API calls (section 9). If retrieval is broken, fix it first.
4. **Run the full evaluation:**

   ```powershell
   uv run rag eval
   ```

   For every question the runner retrieves, generates, judges, checks citations and times each
   stage. About 60 short API calls for 20-30 questions.
5. **Save.** The runner writes `results/v0.1/YYYYMMDD-HHMMSS.json`. It never overwrites a
   file.
6. **Read before concluding** (section 9): failures by hand, five judge verdicts by hand.
7. **Record.** Add the numbers to the table in `README.md` (final runs only) and to the relevant
   row of `limitations.md`; write the reasoning in `decisions.md` if it led to a decision.
8. **Commit code and results separately**, so the history shows what changed and what it
   measured.

### 6.3 Running an experiment

An experiment compares exactly two runs that differ in one variable.

1. State the hypothesis first, in a sentence with a number to beat: "Sentence-aware chunking
   raises `hit@5` on the 11 PDFs".
2. Run the **baseline** with the current code and keep its result file.
3. Change one variable. Re-ingest if the change affects the index (chunking, embedding model,
   query prefix does not).
4. Run again with an identical command.
5. Compare the two result files (section 8). Write the outcome, including when the hypothesis
   was wrong.
6. Keep both files. Do not merge or average runs.

Experiments planned for v0.1:

| # | Variable | Values | Primary metric | Where recorded |
|---|---|---|---|---|
| E1 | Query prefix of bge (`use_query_prefix`) | on / off | `hit@5`, `mrr` | `decisions.md` |
| E2 | Chunking | paragraph-aware (baseline) / sentence-aware | `hit@5`, `mrr`, `correctness` | `limitations.md` L7, README table |
| E3 | Exact terms (no variable, a breakdown) | `exact_term` vs. the rest | `hit@5` per tag | `limitations.md` L1 |
| E4 | Reindex cost (no variable, a timing) | full `ingest` vs. nothing changed | seconds | `limitations.md` L4 |

E1 and E2 only need `--retrieval-only`; run the full evaluation on E2 because chunking also
changes what the generator sees.

### 6.4 Order of work

Following D-012, the pipeline comes before the metrics, and the dataset is written once the
corpus is closed:

1. Embeddings, store, `rag ingest`, `rag ask` (guide steps 5, 6, 7, 9).
2. Close the corpus, write and test the dataset.
3. Retrieval and metrics, then `rag eval --retrieval-only` (step 8).
4. Judge and full evaluation (step 10).
5. Experiments E1-E4 and failure analysis (steps 11-13).

---

## 7. What a run records

One JSON file per run, `results/<version>/<YYYYMMDD-HHMMSS>.json`:

```text
{
  "run": {
    "timestamp": "...",
    "git_commit": "...",              (open: not in the original plan)
    "git_dirty": false,               (open)
    "kind": "retrieval_only" | "full",
    "notes": ""
  },
  "config": { settings without anthropic_api_key, plus prompt file names },
  "dataset": { "path": "...", "n_questions": 24, "n_answerable": 20, "n_unanswerable": 4,
               "sha256": "..." },
  "metrics": {
    "overall": { "hit@1": ..., "hit@5": ..., "mrr": ..., "faithfulness": ...,
                 "correctness": ..., "citation_validity": ...,
                 "abstention_accuracy": ..., "false_abstention": ...,
                 "latency_ms_p50": ..., "latency_ms_p95": ...,
                 "judge_parse_errors": 0 },
    "counts":  { "hit@5": "17/20", ... },
    "by_tag":  { "factual": {...}, "exact_term": {...}, "multi_hop": {...},
                 "unanswerable": {...} }
  },
  "per_question": [
    { "id": "q01", "tags": [...], "retrieved_ids": [...], "rank_of_first_relevant": 2,
      "hit@1": false, "hit@5": true, "rr": 0.5,
      "answer": "...", "cited_ids": [...], "citations_valid": true,
      "verdict": { "faithful": true, "correct": true, "abstained": false, "notes": "..." },
      "latency_ms": { "retrieval": 12, "generation": 840, "total": 861 } }
  ]
}
```

Rules:

- The API key is excluded with `settings.model_dump(mode="json", exclude={"anthropic_api_key"})`.
  Results are committed to a public repository.
- `per_question` stores the answer, the verdict and the cited ids: failure analysis needs them
  and re-running costs money.
- The dataset hash lets a reader detect that two runs used different questions.

---

## 8. Comparing two runs

1. **Same dataset hash and same `n_questions`.** Otherwise the comparison is invalid.
2. **Only the intended variable differs in `config`.** A script (or a manual diff) of the two
   `config` blocks must show one difference.
3. **Compare counts, not only percentages.** Report as "17/20 -> 19/20 (+2 questions)".
4. **Look at the questions that flipped**, in both directions. A gain of 2 with 1 loss is a
   net +1, and the loss may reveal a side effect.
5. **Check the per-tag table.** An overall gain can hide a regression on `exact_term`.
6. **Decision rule.** With this dataset size, treat a change of fewer than 2 questions in a
   metric as "no measurable difference". Do not claim an improvement from it.
7. **Write the result in one sentence with the numbers**, following the style in
   `limitations.md`: "`hit@5` was 0.70 (14/20); sentence-aware chunking raised it to 0.85
   (17/20) (results/v0.1/...)". Never "added X because it is good practice".

The version table in the root README only shows **final runs**, one row per version or
configuration, each linking to its result file.

---

## 9. Quality checks on the evaluation itself

An evaluation can be wrong without the code being wrong. Checks to do each time:

**Retrieval failures.** For every question with `hit@5 = 0`, read the five retrieved chunks and
the evidence. The cause is usually one of:

| Cause | Sign | Action |
|---|---|---|
| Evidence split across two chunks | Both neighbours contain half of it | Chunking issue (real failure), or shorten the evidence |
| Evidence has a word split by the extractor | Normalised text differs by a missing space | Fix the evidence to match `pypdf` output, or note L7 |
| Another document says the same thing | A near-duplicate chunk ranks first | Dataset issue: make the question more specific |
| Genuine semantic miss | Chunks are unrelated | Real retrieval failure |
| Exact term not matched | Acronym or number question | Evidence for L1 |

Only the last two are retrieval failures. Fix the dataset for the others, which counts as a
dataset change (principle 1).

**Judge validation.** After the first full run, read **5 verdicts** by hand (mix correct,
incorrect and abstentions) and note agreement. If you disagree often, change the rubric in a new
file (`judge_v2.txt`; never edit v1) and re-judge. The agreement count is recorded in the notes
of the run.

*(open)* A stronger check: hand-label all answerable questions once and report judge-human
agreement.

**Parse errors.** A judge answer that cannot be parsed becomes an all-false verdict with the
note `judge parse error` and is counted in `judge_parse_errors`. If it is not zero, the
generation metrics of that run are lower bounds; re-run before drawing conclusions.

**Sanity floor.** Two cheap controls that must hold, otherwise the pipeline is broken:

- Asking an unanswerable question returns the abstention sentence.
- Citation validity is 100% on a hand-written cited answer, and 0% when the id is invented
  (unit tests in `tests/test_generation.py`).

---

## 10. Reproducibility and its limits

- **Retrieval is deterministic** for a given index, model and query. Two retrieval-only runs
  must give identical metrics (apart from latency). If they do not, something changed.
- **Generation is not deterministic.** `temperature` is not pinned, so the same run can differ
  by a question or two, possibly more. For that reason differences smaller than 2 questions
  are not claimed (section 8), and repeating the final run is more valuable (see below).
- *(open)* Repeat the final run 3 times and report the range, or accept the single run with the
  caveat above.
- Embedding models are pinned by name in the config; the lockfile pins the library versions.
- Latency depends on the machine and the network (API calls). Compare latencies only between
  runs on the same machine, and report the hardware in the final run's notes.

---

## 11. Limits of this evaluation

State these next to any published number:

- **Small sample.** 20-30 questions: one question moves a rate by 3-5 points, and there are no
  confidence intervals. Percentages are paired with counts for this reason.
- **A single author.** The questions come from one person, and `gold_evidence` was chosen by
  hand, so it reflects one view of "the right chunk".
- **The judge is an LLM.** It can be wrong or biased towards fluent answers; it is checked by
  hand on a sample (section 9), not proven.
- **Strict evidence matching** punishes a correct retrieval whose chunk boundary falls inside
  the evidence, and rewards a chunk that contains the phrase without being useful.
- **Document-level fallback** is looser than evidence-level matching, so questions without
  `gold_evidence` give optimistic retrieval numbers.
- **Closed corpus of 11 regulatory PDFs.** Results say nothing about other domains or about much
  larger corpora (L3).
- **`pypdf` text quality** (L7) affects both retrieval and the evidence strings.
- **No test of out-of-scope robustness beyond 3-5 unanswerable questions**, so
  `abstention_accuracy` is coarse.

---

## 12. Checklist for a run

Copy into the run notes:

- [ ] Corpus and dataset unchanged since the last comparison (hash matches), or both runs
      re-done
- [ ] `rag ingest` reports "up to date"
- [ ] `--retrieval-only` run read; failed questions inspected
- [ ] Only one variable differs from the baseline
- [ ] Full run saved under a new timestamped file
- [ ] `judge_parse_errors` is 0
- [ ] Five judge verdicts checked by hand
- [ ] Counts, per-tag table and flipped questions compared
- [ ] Number written in `limitations.md` / `decisions.md` / README table as appropriate
- [ ] Code and results committed separately

---

## 13. Open points for review

1. **Run metadata:** add `git_commit`, `git_dirty` and the dataset hash to the result file
   (section 7). Not in the original guide.
2. **Final-run generator:** keep Haiku for every v0.1 run (proposal) or use Sonnet for the final
   run (section 5).
3. **Citation validity denominator:** exclude abstentions (proposal, section 4.4).
4. **Repeat the final run** several times to measure generation noise (section 10).
5. **Judge-human agreement:** 5 spot checks (original plan) or full hand labelling (section 9).
6. **Decision threshold:** "fewer than 2 questions is no difference" (section 8) is a rule of
   thumb; confirm or replace it with a proper test once the dataset size is known.
7. **`gold_evidence` multi-chunk case:** `multi_hop` questions may need evidence from two
   places. The format has one `gold_evidence` field. Either those questions use the document
   fallback, or the field becomes a list (a change to `EvalCase` and the dataset format).
