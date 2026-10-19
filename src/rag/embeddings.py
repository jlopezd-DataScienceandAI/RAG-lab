"""Text to vectors. Everything else talks to this class, so swapping models is a one-file change."""

import numpy as np


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None  # loaded lazily so that importing the package stays fast

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Embed chunk texts. Returns a float32 array of shape [len(texts), dim].

        TODO(v0.1):
          - load sentence-transformers lazily on first use
          - L2-normalise the vectors, so a dot product equals cosine similarity
        """
        raise NotImplementedError("encode_passages")

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a question. Returns a float32 array of shape [dim].

        bge models recommend a query instruction prefix for retrieval; check the model card
        and measure with and without it (a nice first experiment).
        """
        raise NotImplementedError("encode_query")
