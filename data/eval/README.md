# Evaluation dataset

`questions.jsonl` holds one JSON object per line (see `EvalCase` in `src/rag/models.py`).
The file does not exist yet: it is written by hand once the pipeline works.

| Field | Meaning |
|---|---|
| `id` | Unique id, e.g. `q01` |
| `question` | The question as a user would ask it |
| `answerable` | `false` for questions the corpus cannot answer (tests abstention) |
| `gold_doc` | File that contains the answer (required when answerable) |
| `gold_section` | Optional section or page that narrows it down |
| `gold_evidence` | Short literal phrase (5-15 words) copied from the source, which must appear in the correct chunk |
| `expected_answer` | Short reference answer used by the judge |
| `tags` | Question types (see below), used to report metrics per type |

Question types:

| Type | How many | Tag | Example |
|---|---|---|---|
| Direct fact | 10-15 | `factual` | "What does X default to?" |
| Exact term, code, acronym or name | 4-6 | `exact_term` | "What is error E1042?" |
| Needs two parts joined | 2-4 | `multi_hop` | "Which of A or B supports C?" |
| No answer in the corpus | 3-5 | `unanswerable` | Plausible, but not in the documents |

Guidelines:

- Phrase questions in your own words, not by copying sentences from the documents.
- `gold_evidence` must be literal: copy and paste it from the document.
- For unanswerable questions set `"answerable": false` and `null` for `gold_doc`,
  `gold_evidence` and `expected_answer`.

- Write 20-30 questions by hand while reading the documents, **before** writing the pipeline.
- Include 3-5 unanswerable questions.
- Include some questions that need exact terms (codes, acronyms, names): they expose the
  weakness of pure vector search and motivate BM25 in v0.3.
- Every version is evaluated against this same file. If you change it, note it in
  `CHANGELOG.md` and re-run the earlier versions.
