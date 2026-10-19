import pytest

from rag.store.base import VectorStore
from rag.store.numpy_store import NumpyStore


def test_interface_cannot_be_instantiated():
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


def test_numpy_store_starts_empty():
    assert len(NumpyStore()) == 0
