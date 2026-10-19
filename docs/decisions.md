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
