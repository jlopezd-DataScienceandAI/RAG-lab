"""Exact cosine search over a numpy matrix, with chunk metadata in SQLite.

On disk (see docs/decisions.md, D-003):

    index/chunks.db    SQLite table ``chunks``; row_idx is the row in vectors.npy
    index/vectors.npy  float32 matrix [n_chunks, dim], same order as row_idx
    index/meta.json    embedding model, chunk settings, n_chunks, creation date
"""

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag.models import Chunk, RetrievalResult
from rag.store.base import VectorStore

SCHEMA = """
CREATE TABLE chunks (
    row_idx   INTEGER PRIMARY KEY,
    id        TEXT UNIQUE NOT NULL,
    doc       TEXT NOT NULL,
    page      INTEGER,
    section   TEXT,
    text      TEXT NOT NULL,
    n_tokens  INTEGER NOT NULL,
    doc_hash  TEXT NOT NULL
)
"""

_OUT_OF_SYNC = "index is out of sync: rebuild it with `rag ingest`"


class NumpyStore(VectorStore):
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Append chunks and their vectors; row i of ``vectors`` belongs to ``chunks[i]``.

        All checks run before anything is modified, so a failed call leaves the store as it was.
        """
        vectors = np.asarray(vectors)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D [n_chunks, dim], got shape {vectors.shape}")
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"got {len(chunks)} chunks but {vectors.shape[0]} vectors: they must match"
            )
        if len(chunks) == 0:
            return
        if self._vectors is not None and vectors.shape[1] != self._vectors.shape[1]:
            raise ValueError(
                f"vector dimension {vectors.shape[1]} does not match the store's "
                f"{self._vectors.shape[1]}"
            )

        new = np.array(vectors, dtype=np.float32, copy=True)  # copy: don't alias the caller's array
        self._vectors = new if self._vectors is None else np.vstack([self._vectors, new])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievalResult]:
        """Return the ``k`` most similar chunks, best first, with 1-based ranks.

        The score is a dot product, which equals cosine similarity because the embedder
        normalizes its vectors.
        """
        if self._vectors is None or not self._chunks or k < 1:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (self._vectors.shape[1],):
            raise ValueError(
                f"query vector has shape {query.shape}, expected ({self._vectors.shape[1]},)"
            )

        scores = self._vectors @ query
        k = min(k, len(self._chunks))
        top = np.argpartition(-scores, k - 1)[:k]  # the k best, unordered
        top = top[np.argsort(-scores[top])]  # best to worst
        return [
            RetrievalResult(chunk=self._chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(top, start=1)
        ]

    def save(self, directory: Path, extra_meta: dict | None = None) -> None:
        """Write vectors.npy, chunks.db and meta.json.

        Everything is written to temporary files first and renamed at the end, so an interrupted
        run cannot leave chunks.db and vectors.npy out of sync (limitation L5).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        tmp_vectors = directory / "vectors.tmp.npy"  # must end in .npy or np.save adds it
        tmp_db = directory / "chunks.tmp.db"
        tmp_meta = directory / "meta.tmp.json"
        tmp_files = (tmp_vectors, tmp_db, tmp_meta)

        dim = int(self._vectors.shape[1]) if self._vectors is not None else 0
        matrix = self._vectors if self._vectors is not None else np.empty((0, 0), dtype=np.float32)
        meta = {
            "n_chunks": len(self._chunks),
            "dim": dim,
            "created_at": datetime.now(UTC).isoformat(),
            **(extra_meta or {}),
        }

        try:
            for tmp in tmp_files:  # leftovers from an interrupted run
                tmp.unlink(missing_ok=True)

            np.save(tmp_vectors, matrix)

            rows = [
                (i, c.id, c.doc, c.page, c.section, c.text, c.n_tokens, c.doc_hash)
                for i, c in enumerate(self._chunks)
            ]
            con = sqlite3.connect(tmp_db)
            try:
                con.execute(SCHEMA)
                con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
                con.commit()
            finally:
                con.close()  # Windows cannot rename an open file

            tmp_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

            os.replace(tmp_vectors, directory / "vectors.npy")
            os.replace(tmp_db, directory / "chunks.db")
            os.replace(tmp_meta, directory / "meta.json")
        finally:
            for tmp in tmp_files:  # nothing is left behind, even if something failed
                tmp.unlink(missing_ok=True)

    def load(self, directory: Path) -> None:
        """Load a saved store and check that its three files agree with each other.

        The store is replaced only if every check passes.
        """
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npy")

        db_path = directory / "chunks.db"
        if not db_path.exists():  # sqlite3.connect would silently create an empty file
            raise FileNotFoundError(db_path)
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(
                "SELECT row_idx, id, doc, page, section, text, n_tokens, doc_hash "
                "FROM chunks ORDER BY row_idx"  # without ORDER BY the order is not guaranteed
            ).fetchall()
        finally:
            con.close()

        n = len(rows)
        if (
            vectors.ndim != 2
            or n != vectors.shape[0]
            or n != meta.get("n_chunks")
            or [r[0] for r in rows] != list(range(n))
        ):
            raise RuntimeError(_OUT_OF_SYNC)

        self._chunks = [
            Chunk(
                id=r[1], doc=r[2], page=r[3], section=r[4], text=r[5], n_tokens=r[6], doc_hash=r[7]
            )
            for r in rows
        ]
        self._vectors = vectors.astype(np.float32, copy=False) if n else None

    def __len__(self) -> int:
        return len(self._chunks)
