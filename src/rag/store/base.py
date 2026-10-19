"""The contract every vector store implements.

The rest of the code only talks to this interface, so moving from numpy to FAISS, Chroma,
Qdrant or pgvector in v0.3 does not touch retrieval, generation or evaluation.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from rag.models import Chunk, RetrievalResult


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Add chunks and their vectors (row i of ``vectors`` belongs to ``chunks[i]``)."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievalResult]:
        """Return the ``k`` most similar chunks, best first, with 1-based ranks."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persist the store to disk."""

    @abstractmethod
    def load(self, directory: Path) -> None:
        """Load a previously saved store."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of chunks in the store."""
