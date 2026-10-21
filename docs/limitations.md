# Limitations of v0.1 and the experiments that measure them

Each limitation is a hypothesis. Measure it in v0.1, record the numbers in `results/v0.1/`,
and let the numbers justify the change in the next version.

| ID | Expected limitation | How to measure it | Measured | Motivates |
|---|---|---|---|---|
| L1 | Dense search misses exact terms, codes, acronyms and names | Hit@k on the dataset questions that need literal terms vs. the rest | TBD | v0.3: BM25 + hybrid search |
| L2 | The right chunk is retrieved but not ranked first | Hit@5 high and MRR low | TBD | v0.3: reranking |
| L3 | Exact search compares against every vector | Duplicate vectors synthetically (10k, 100k, 1M); measure latency and RAM | TBD | Vector database or approximate index |
| L4 | Any document change means reindexing everything | Time a full `ingest` vs. an incremental one | TBD | Incremental updates, pgvector |
| L5 | `chunks.db` and `vectors.npy` can get out of sync | Test that interrupts a save and checks consistency on load | TBD | A single transactional store |
| L6 | No interface, no concurrent access | Evident | n/a | v0.2: API |
| L7 | `pypdf` rarely emits blank lines, so a PDF page becomes one "paragraph" and paragraph-aware chunking falls back to cutting by token windows, mid-sentence | Share of chunks that start in lowercase or end without punctuation, per document; re-measure with sentence-aware chunking (same corpus, dataset, model, size and overlap) | Baseline chunker (500 tokens, 75 overlap): `EU_GDPR.pdf` has 0 of 78 pages with a blank line; 46 of 141 chunks start in lowercase and 132 of 141 end without punctuation. `EBA_GL_Internal_Governance.pdf` has 79 of 93 pages with a blank line; 54 of 163 and 105 of 163 | v0.1 step 11: sentence-aware chunking; later, layout-aware PDF extraction |

## How to fill this in

1. Run the experiment and save the output under `results/v0.1/`.
2. Write the number in the *Measured* column and link the results file.
3. When a later version fixes the limitation, add the before/after numbers here.

Example of the target sentence: "MRR was 0.41; adding a cross-encoder reranker raised it to
0.63 (results/v0.3/...)". Never "added reranking because it is good practice".
