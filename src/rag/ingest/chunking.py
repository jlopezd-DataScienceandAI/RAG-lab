"""Split pages into overlapping chunks. This is where most retrieval quality is won or lost."""

from rag.ingest.loaders import RawPage
from rag.models import Chunk


def chunk_pages(pages: list[RawPage], max_tokens: int, overlap: int) -> list[Chunk]:
    """Turn pages into chunks of at most ``max_tokens`` tokens with ``overlap`` shared tokens.

    TODO(v0.1):
      - count tokens with tiktoken (an approximation of the embedding model's tokenizer)
      - split on paragraph boundaries first, and only cut inside a paragraph if it is too long
      - build ids like ``{doc}::p{page}::c{index}`` and fill n_tokens and doc_hash
      - read the chunks by eye afterwards: are sentences cut? are headings lost?

    The invariants in tests/test_chunking.py describe what "correct" means.
    """
    raise NotImplementedError("chunk_pages")
