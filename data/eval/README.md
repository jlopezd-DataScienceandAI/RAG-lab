# Evaluation dataset

`questions.jsonl` is the ground truth every version is evaluated against. It holds one JSON
object per line (see `EvalCase` in `src/rag/models.py`). This file only describes the
**format**; how the dataset is used, which metrics are computed and how runs are compared is in
[`docs/evaluation.md`](../../docs/evaluation.md). It does not exist yet.

## Format

Answerable question (`factual`, `exact_term`, `multi_hop`):

```json
{"id": "q01", "question": "How must high-risk AI systems allow human oversight?", "answerable": true, "gold_doc": "EU_AI_Act.pdf", "gold_evidence": "can be effectively overseen by natural persons", "expected_answer": "They must be designed so natural persons can effectively oversee them.", "tags": ["factual"]}
```

Unanswerable question:

```json
{"id": "q21", "question": "What penalty does the Spanish consumer code set for late repayment?", "answerable": false, "gold_doc": null, "gold_evidence": null, "expected_answer": null, "tags": ["unanswerable"]}
```

One object per line, no line breaks inside it, always with the same seven keys in this order.

| Field | Meaning |
|---|---|
| `id` | Unique id: `q01`, `q02`, ... |
| `question` | The question as a user would ask it, in your own words |
| `answerable` | `false` for questions the corpus cannot answer (tests abstention) |
| `gold_doc` | Exact file name in `data/raw/` that contains the answer (required when answerable) |
| `gold_evidence` | Literal phrase (5-15 words) copied from the source; it must appear in the correct chunk |
| `expected_answer` | Short reference answer used by the judge: a figure, a name or one sentence |
| `tags` | A list with the question type (see below), used to report metrics per type |

Unanswerable questions have `gold_doc`, `gold_evidence` and `expected_answer` set to `null`.
`EvalCase` only enforces `gold_doc` for answerable questions; the rest is checked by
`tests/test_smoke.py`.

## Question types

| Type | How many | Tag | Example |
|---|---|---|---|
| Direct fact | 11-15 | `factual` | "What does the AI Act say about X?" |
| Exact term, acronym, number or name | 4-6 | `exact_term` | "What does LGD stand for?" |
| Needs two parts joined | 2-4 | `multi_hop` | "Which of A or B covers C?" |
| No answer in the corpus | 3-5 | `unanswerable` | Plausible, but not in the documents |

The minimums add up to 20, the maximums to 30.

## Guidelines

- Phrase questions in your own words, not by copying sentences from the documents.
- `gold_evidence` is literal: copy it from the text **as `pypdf` extracts it**, not from the PDF
  viewer. Relevance normalises case and whitespace, but not words split by the extractor
  (see L7).
- Write the questions while reading the documents, never after looking at what the retriever
  returns.
- Changing this file has consequences for every earlier result; see
  [`docs/evaluation.md`](../../docs/evaluation.md).
