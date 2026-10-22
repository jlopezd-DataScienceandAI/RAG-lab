"""Retrieval metrics, implemented by hand on purpose (ragas comes in v0.3)."""

from rag.models import EvalCase, RetrievalResult


def is_relevant(result: RetrievalResult, case: EvalCase) -> bool:
    """TODO(v0.1): relevant if gold_evidence is in the chunk text, else if doc == gold_doc."""
    raise NotImplementedError("is_relevant")


def hit_at_k(results: list[RetrievalResult], case: EvalCase, k: int) -> bool:
    """True if a relevant chunk appears among the first ``k`` results."""
    raise NotImplementedError("hit_at_k")


def reciprocal_rank(results: list[RetrievalResult], case: EvalCase) -> float:
    """1 / rank of the first relevant chunk, or 0.0 if there is none. MRR is the mean."""
    raise NotImplementedError("reciprocal_rank")
