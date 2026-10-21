"""Text to vectors. Everything else talks to this class, so swapping models is a one-file change."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# bge models were trained with this instruction in front of the question, and without it in
# front of the passages (see the model card).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32


class Embedder:
    def __init__(self, model_name: str, use_query_prefix: bool = True) -> None:
        self.model_name = model_name
        self.use_query_prefix = use_query_prefix
        self._model: SentenceTransformer | None = None  # loaded lazily: keeps `import rag` fast

    def _get_model(self) -> "SentenceTransformer":
        """Load the model on first use and reuse it afterwards."""
        if self._model is None:
            # Imported here because sentence-transformers (and torch) take seconds to import.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Embed chunk texts. Returns a float32 array of shape [len(texts), dim].

        Vectors are L2-normalised, so a dot product between two of them equals their cosine
        similarity and the store does not need to compute norms at search time.
        """
        model = self._get_model()
        if not texts:
            # The model does not return a 2D array for an empty list, so build it explicitly
            dim = model.get_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > BATCH_SIZE,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a question. Returns a float32 array of shape [dim].

        bge models recommend a query instruction prefix for retrieval; it is on by default and
        can be switched off with ``use_query_prefix=False`` to measure its effect (step 8).
        """
        prefix = QUERY_PREFIX if self.use_query_prefix else ""
        vectors = self._get_model().encode(
            [prefix + text],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors[0], dtype=np.float32)  # [1, dim] -> [dim]
