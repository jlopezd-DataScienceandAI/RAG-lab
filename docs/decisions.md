# Design decisions

One entry per decision: what was chosen, why, what else was considered, and when to revisit.

## D-001 No orchestration frameworks in v0.1 (2026-10-19)
- **Decision:** write chunking, retrieval, prompting and evaluation by hand.
- **Why:** frameworks hide exactly the parts that determine quality; writing them is how I
  learn what each one does and where it fails.
- **Alternatives:** LangChain, LlamaIndex.
- **Revisit:** optionally in v0.3 as a comparison, not as a replacement.

## D-002 English corpus, local embeddings (2026-10-19)
- **Decision:** English documents; `BAAI/bge-small-en-v1.5` through sentence-transformers.
- **Why:** best-supported language; local models make the repo reproducible without extra
  accounts or cost. Anthropic offers no embedding model of its own.
- **Alternatives:** `fastembed` (lighter, no PyTorch), a hosted embedding API.
- **Revisit:** v0.3, comparing larger or different models on the same dataset.

## D-003 numpy + SQLite behind a `VectorStore` interface (2026-10-19)
- **Decision:** vectors in `index/vectors.npy`, chunks and metadata in `index/chunks.db`,
  row `i` of the matrix corresponds to `row_idx = i` in the table.
- **Why:** zero extra dependencies, every file can be inspected by hand, exact search, and I
  learn how persistence and search work underneath.
- **Alternatives:** Chroma, FAISS, Qdrant, pgvector.
- **Known limits:** see `docs/limitations.md` (L3, L4, L5).
- **Revisit:** v0.3, compare other stores on the same dataset.

## D-004 pypdf for PDF extraction (2026-10-19)
- **Decision:** `pypdf`, page by page.
- **Why:** permissive licence (BSD), gives page numbers for citations.
- **Alternatives:** PyMuPDF (better with tables and complex layout, AGPL licence).
- **Revisit:** if the corpus has heavily laid-out PDFs.

## D-005 Claude Haiku 5.5 to iterate, Sonnet 5.5 for final numbers (2026-10-19)
- **Decision:** `claude-haiku-5-5` for generation while developing, `claude-sonnet-5-5` as the
  judge and for the runs that go into the README. Both are configurable.
- **Why:** cheap iteration, stronger judge for the numbers that get published.

## D-006 Evaluation written by hand before ragas (2026-10-19)
- **Decision:** implement Hit@k, MRR, a rubric-based judge and a citation check myself.
- **Why:** understand what each metric measures before delegating it to a library.
- **Revisit:** v0.3, add ragas and compare against the hand-written metrics.

## D-007 One code base, tags for versions (2026-10-19)
- **Decision:** `main` always holds the latest code; each version is developed on a branch,
  merged through a pull request, and marked with a tag and a GitHub Release.
- **Why:** git already records history; folders per version duplicate code and hide it.

## D-008 Paragraph-aware token chunking, 500 tokens / 75 overlap (2026-10-20)
- **Decision:** split each page on blank lines; glue short heading-like paragraphs (under 12
  tokens, no final punctuation) to the next one; pack paragraphs into chunks of at most 500
  tokens; cut a paragraph longer than that with a sliding window; repeat the last 75 tokens
  at the start of the next chunk.
- **Why:** paragraphs are the natural unit of legal and regulatory text; counting tokens makes
  sizes comparable across documents; a heading alone makes a useless chunk. The 500/75 values
  are the starting point of the v0.1 guide (about 15% overlap), not tuned yet.
- **Known limit:** window cuts and the overlap work on tokens, so a sentence can be cut in
  half. This is the baseline that the sentence-aware strategy (guide, step 11) is measured
  against.
- **Alternatives:** fixed-size character windows, sentence-aware packing, splitting by
  article or section.
- **Revisit:** step 11 (paragraph vs. sentence comparison) and later chunk-size experiments.

## D-009 tiktoken `cl100k_base` to count tokens (2026-10-20)
- **Decision:** measure chunk size with tiktoken (`cl100k_base`), not with the embedding
  model's own tokenizer.
- **Why:** fast, no model needed at chunking time, and good enough to compare sizes.
- **Known risk (to verify):** `bge-small-en-v1.5` has its own tokenizer and a 512-token
  input limit. A chunk of 500 `cl100k` tokens may exceed 512 model tokens and be truncated
  silently. Before step 8, count how many chunks go over the limit with the model tokenizer.
- **Alternatives:** count with the embedding model's tokenizer.
- **Revisit:** if the check above finds truncated chunks, or when changing the embedding model.

## D-010 Chunk ids and page handling (2026-10-20)
- **Decision:** the id is `{doc}::p{page}::c{index:03d}`, where `doc` is the file name, `page`
  is the 1-based PDF page (0 for formats without pages) and `index` counts chunks per
  document without resetting between pages. Chunks never cross a page boundary. `section`
  stays `None` in v0.1.
- **Why:** readable ids that the model can cite and a person can trace back to a page; ids do
  not depend on where the repository lives.
- **Known limit:** a paragraph that continues on the next page is split in two; PDF headers,
  footers and page numbers are not removed.
- **Revisit:** if the failure analysis (guide, step 12) points to these cuts.

## D-011 Evaluation set written by hand, with unanswerable questions (2026-10-19)
- **Decision:** 17 questions written from the four PDFs in `data/raw/` (AI Act, GDPR, the
  default-definition guidelines and the PD/LGD estimation guidelines): 14 answerable and 3
  unanswerable (they ask about national rules that are not in the corpus), to test that the
  system abstains.
- **Why:** the questions come before the pipeline results so they are not biased towards what
  the retriever happens to find.
- **Status:** provisional. It is extended to 20-30 questions, with `gold_evidence` and
  `tags`, once the corpus is closed (see D-012).

## D-012 Pipeline first, evaluation after the corpus and dataset are frozen (2026-10-20)
- **Decision:** implement embeddings, store, `rag ingest` and `rag ask` (guide steps 5, 6, 7
  and 9) on the current corpus; then close the corpus and finish the dataset; only then
  implement metrics and the full evaluation (steps 8 and 10).
- **Why:** more documents add distractors and change the difficulty, so a measurement made
  earlier would have to be repeated. The new questions are written while reading the
  documents, never after looking at what the retriever returns.
- **Alternatives:** follow the guide order literally (metrics at step 8 on the current corpus).
- **Revisit:** when the corpus is closed.

## D-013 One source for the build order (2026-10-20)
- **Decision:** `docs/v0.1-guide.md` is the only place that lists the steps of v0.1;
  `docs/workflow.md` links to it and covers only git and experiments.
- **Why:** two copies of the plan drifted apart (13 steps vs. 15).

## D-014 Project name: rag-lab (2026-10-20)
- **Decision:** the project is called `rag-lab` in the README, `pyproject.toml`, the lockfile
  and the docs, matching the GitHub repository.
- **Why:** the docs used `rag-lab` and the packaging used `rag-project`.

## D-015 Add `fonttools` so pypdf can decode CFF Type1 fonts (2026-10-22)
- **Decision:** add `fonttools` as a direct dependency.
- **Why:** pypdf warns "fontTools is required to fully parse the encoding of a CFF Type1 font"
  on `AESIA_AI_Act_Practical_Guides.pdf` (Gotham fonts). The extracted text was checked and is
  valid for that file (no U+FFFD), because those fonts declare `/WinAnsiEncoding`. Other PDFs
  with custom encodings could silently produce wrong text, and the warning is noise that hides
  real problems.
- **Alternatives:** silence the `pypdf` logger (no new dependency, but hides a warning that
  can be real in another PDF).
- **Cost:** one small pure-Python dependency, already a transitive tool in the PDF ecosystem.
- **Not fixed by this:** words split by layout (`Int\nroductory`, `Guid\ne 1`) come from pypdf
  text extraction, not from fonts (see L7).
