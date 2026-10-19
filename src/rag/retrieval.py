"""Question in, ranked chunks out. No LLM involved, so it can be evaluated on its own."""

from rag.embeddings import Embedder
from rag.models import RetrievalResult
from rag.store.base import VectorStore


def retrieve(
    question: str, embedder: Embedder, store: VectorStore, k: int
) -> list[RetrievalResult]:
    """TODO(v0.1): embed the question and return the top-k results from the store."""
    raise NotImplementedError("retrieve")
