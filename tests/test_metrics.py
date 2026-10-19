"""Executable spec for the retrieval metrics. Remove the xfail marker once implemented."""

import pytest

from rag.evaluation.metrics import hit_at_k, reciprocal_rank
from rag.models import Chunk, EvalCase, RetrievalResult

IMPLEMENT = pytest.mark.xfail(raises=NotImplementedError, reason="implement the metrics first")


def _result(doc: str, rank: int) -> RetrievalResult:
    chunk = Chunk(id=f"{doc}::c{rank}", doc=doc, text="x", n_tokens=1, doc_hash="h")
    return RetrievalResult(chunk=chunk, score=1.0 / rank, rank=rank)


CASE = EvalCase(id="q1", question="?", gold_doc="right.pdf")


@IMPLEMENT
def test_hit_at_k_depends_on_k():
    results = [_result("other.pdf", 1), _result("right.pdf", 2)]
    assert hit_at_k(results, CASE, k=2)
    assert not hit_at_k(results, CASE, k=1)


@IMPLEMENT
def test_reciprocal_rank():
    results = [_result("other.pdf", 1), _result("right.pdf", 2)]
    assert reciprocal_rank(results, CASE) == 0.5
    assert reciprocal_rank([_result("other.pdf", 1)], CASE) == 0.0
