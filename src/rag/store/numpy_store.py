"""Exact cosine search over a numpy matrix, with chunk metadata in SQLite.

On disk (see docs/decisions.md, D-003):

    index/chunks.db    SQLite table ``chunks``; row_idx is the row in vectors.npy
    index/vectors.npy  float32 matrix [n_chunks, dim], same order as row_idx
    index/meta.json    embedding model, chunk settings, n_chunks, creation date
"""

from pathlib import Path

import numpy as np

from rag.models import Chunk, RetrievalResult
from rag.store.base import VectorStore


class NumpyStore(VectorStore):
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """TODO(v0.1): check shapes match, then append to the in-memory matrix and list."""
        raise NotImplementedError("add")

    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievalResult]:
        """TODO(v0.1): scores = vectors @ query_vector, take the top k, build results.

        Hint: np.argpartition is faster than a full sort when k is much smaller than n.
        """
        raise NotImplementedError("search")

    def save(self, directory: Path) -> None:
        """TODO(v0.1): write vectors.npy, chunks.db and meta.json.

        Write to temporary files and rename them at the end, so an interrupted run cannot
        leave chunks.db and vectors.npy out of sync (this is limitation L5).
        """
        raise NotImplementedError("save")

    def load(self, directory: Path) -> None:
        """TODO(v0.1): load both files and check that n_chunks in meta.json matches."""
        raise NotImplementedError("load")

    def __len__(self) -> int:
        return len(self._chunks)
