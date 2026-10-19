# Evaluation dataset

`questions.jsonl` holds one JSON object per line (see `EvalCase` in `src/rag/models.py`).
The three lines in it now are placeholders: replace them with your own questions.

| Field | Meaning |
|---|---|
| `id` | Unique id, e.g. `q01` |
| `question` | The question as a user would ask it |
| `answerable` | `false` for questions the corpus cannot answer (tests abstention) |
| `gold_doc` | File that contains the answer (required when answerable) |
| `gold_section` | Optional section or page that narrows it down |
| `expected_answer` | Short reference answer used by the judge |

Guidelines:

- Write 20-30 questions by hand while reading the documents, **before** writing the pipeline.
- Include 3-5 unanswerable questions.
- Include some questions that need exact terms (codes, acronyms, names): they expose the
  weakness of pure vector search and motivate BM25 in v0.3.
- Every version is evaluated against this same file. If you change it, note it in
  `CHANGELOG.md` and re-run the earlier versions.
