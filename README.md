# rag-lab

A small Retrieval-Augmented Generation (RAG) pipeline built from scratch: no LangChain, no
LlamaIndex, no vector database. Every piece (chunking, retrieval, prompting, evaluation) is
written by hand so that its behaviour and limits are understood, measured, and then improved
one version at a time.

> Status: **v0.1 in progress.** See [CHANGELOG.md](CHANGELOG.md).

## Versions and results

Every version is evaluated on the same dataset (`data/eval/questions.jsonl`).

| Version | Main change | Hit@5 | MRR | Latency | Details |
|---|---|---|---|---|---|
| v0.1 | Baseline: numpy + SQLite, dense retrieval, cited answers, paragraph chunking | TBD | TBD | TBD | TBD |
| v0.1 | Same baseline with sentence-aware chunking (only change) | TBD | TBD | TBD | TBD |

Each limitation found in one version motivates the next one: see
[docs/limitations.md](docs/limitations.md).

## How it works (v0.1)

```
Indexing (rag ingest)
  documents -> chunking -> embeddings -> index/ (SQLite + vectors.npy)

Querying (rag ask)
  question -> embed -> top-k chunks -> prompt -> Claude -> answer with citations

Evaluation (rag eval)
  questions.jsonl -> retrieval metrics (Hit@k, MRR) + judged answers -> results/
```

## Quickstart

```bash
uv sync --extra dev            # or: python -m venv .venv && pip install -e ".[dev]"
# create a .env file in the repo root containing: ANTHROPIC_API_KEY=<your key>
# put your documents in data/raw/
uv run rag ingest
uv run rag ask "What is ...?"
uv run rag eval
```

## Corpus

11 public regulatory and supervisory PDFs in `data/raw/`, all in English, downloaded on
2026-10-20. EUR-Lex and EBA documents are public sources; the licence column is the general
reuse regime of each publisher, not a per-document check.

| File | What it is | Source | Licence / reuse |
|---|---|---|---|
| `EBA_GL_Loan_Origination_and_Monitoring.pdf` | EBA/GL/2020/06, Guidelines on loan origination and monitoring (final report, 29 May 2020, 89 pp.) | [EBA](https://www.eba.europa.eu/regulation-and-policy/credit-risk/guidelines-on-loan-origination-and-monitoring) | EBA legal notice: reuse allowed with source acknowledgement |
| `EU_AI_Act.pdf` | Regulation (EU) 2024/1689, Artificial Intelligence Act (Official Journal, 12 Jul 2024, 144 pp.) | [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/oj?locale=en) | EU reuse policy (Decision 2011/833/EU, CC BY 4.0) |
| `EU_Consumer_Credit_Directive.pdf` | Directive (EU) 2023/2225 on credit agreements for consumers (OJ 30 Oct 2023, 67 pp.) | [EUR-Lex](https://eur-lex.europa.eu/eli/dir/2023/2225/oj) | EU reuse policy (CC BY 4.0) |
| `EU_GDPR.pdf` | Regulation (EU) 2016/679, GDPR, consolidated text (78 pp.). Article 22 (automated decisions) is the key one | [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng) | EU reuse policy (CC BY 4.0). Consolidated text has no legal effect |
| `EBA_ML_for_IRB_Models_Follow_up_Report.pdf` | EBA follow-up report on the machine learning for IRB models discussion paper (Aug 2023, 24 pp.) | [EBA press release](https://www.eba.europa.eu/eba-publishes-follow-report-use-machine-learning-internal-ratings-based-models) | EBA legal notice: reuse allowed with source acknowledgement |
| `AESIA_AI_Act_Practical_Guides.pdf` | Overview of the AESIA (Spanish AI supervisor) guidelines for AI Act compliance (June 2026, 11 pp.). Non-binding | [AESIA](https://aesia.digital.gob.es/en/present/resources/practical-guides-for-ai-act-compliance) / [HSF Kramer](https://www.hsfkramer.com/notes/madrid/2026-posts/full-compilation-aesia-guide-english-june-2026) | Not verified, check before redistributing |
| `EBA_GL_PD_LGD_Estimation.pdf` | EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and treatment of defaulted exposures (20 Nov 2017, 200 pp.) | [EBA](https://www.eba.europa.eu/documents/10180/2033363/6b062012-45d6-4655-af04-801d26493ed0/Guidelines%20on%20PD%20and%20LGD%20estimation%20%28EBA-GL-2017-16%29.pdf) | EBA legal notice: reuse allowed with source acknowledgement |
| `EBA_Big_Data_and_Advanced_Analytics.pdf` | EBA/REP/2020/01, Report on Big Data and Advanced Analytics (Jan 2020, 60 pp.) | [EBA](https://www.eba.europa.eu/sites/default/documents/files/document_library/Final%20Report%20on%20Big%20Data%20and%20Advanced%20Analytics.pdf) | EBA legal notice: reuse allowed with source acknowledgement |
| `Spain_Consumer_Credit_Act.pdf` | Act 16/2011 on Consumer Credit Contracts, English translation (2017 edition, 47 pp.). Scheduled to be repealed | [Ministerio de Justicia](https://www.mjusticia.gob.es/es/AreaTematica/DocumentacionPublicaciones/Documents/Act%2016-2011%2C%20of%2024%20June%2C%20on%20Consumer%20Credit%20Contracts.pdf) | Unofficial translation; Spanish public-sector information reuse regime (Ley 37/2007) |
| `EBA_GL_Internal_Governance.pdf` | EBA/GL/2021/05, Guidelines on internal governance under Directive 2013/36/EU (final report, 2 Jul 2021, 93 pp.) | [EBA](https://www.eba.europa.eu/sites/default/documents/files/document_library/Publications/Guidelines/2021/1016721/Final%20report%20on%20Guidelines%20on%20internal%20governance%20under%20CRD.pdf) | EBA legal notice: reuse allowed with source acknowledgement |
| `EU_CRR.pdf` | Regulation (EU) 575/2013, CRR, consolidated version of 1 Jan 2025 (897 pp.) | [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2013/575/2025-01-01) | EU reuse policy (CC BY 4.0). Consolidated text has no legal effect |

Caveats for citations and evaluation:

- Consolidated EUR-Lex texts (GDPR, CRR) are documentation tools; cite the Official Journal
  version in any written report. The GDPR file is the 2016 consolidated text.
- The Spanish Act is an unofficial translation; the authoritative text is in Spanish at
  [boe.es](https://www.boe.es).
- The AESIA file is an 11-page overview, not the full compilation of the 16 guides.

## Project layout

```
data/raw/            source documents
data/eval/           evaluation questions
index/               generated index (git-ignored)
prompts/             versioned prompt templates
src/rag/             the pipeline (ingest, embeddings, store, retrieval, generation, evaluation)
tests/               unit tests
results/<version>/   evaluation output, kept as historical evidence
docs/                design decisions, limitations, workflow
```

## Roadmap

- **v0.1** baseline RAG with its own evaluation, plus a measured comparison of paragraph vs.
  sentence-aware chunking
- **v0.2** API (FastAPI) and a small demo UI
- **v0.3** hybrid search (BM25 + vectors), reranking, vector-store comparison, ragas
- **v0.4** agent with tools
- **v0.5** Docker, CI, tracing of cost and latency

## Documentation

- [Design decisions](docs/decisions.md)
- [Limitations and the experiments that measure them](docs/limitations.md)
- [Development workflow](docs/workflow.md)
