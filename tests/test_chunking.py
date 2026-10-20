"""Executable spec for chunking. Remove the xfail marker once chunk_pages is implemented."""


from rag.ingest.chunking import chunk_pages
from rag.ingest.loaders import RawPage


def test_chunks_respect_max_tokens():
    pages = [RawPage(doc="a.md", page=None, text="word " * 2000)]
    chunks = chunk_pages(pages, max_tokens=100, overlap=10)
    assert chunks
    assert all(chunk.n_tokens <= 100 for chunk in chunks)


def test_chunk_ids_are_unique_and_readable():
    pages = [RawPage(doc="a.md", page=None, text="Some text. " * 500)]
    chunks = chunk_pages(pages, max_tokens=100, overlap=10)
    ids = [chunk.id for chunk in chunks]
    assert len(set(ids)) == len(ids)
    assert all(chunk_id.startswith("a.md::") for chunk_id in ids)


def test_no_text_is_lost():
    text = " ".join(f"token{i}" for i in range(1000))
    chunks = chunk_pages([RawPage(doc="a.md", page=None, text=text)], max_tokens=100, overlap=10)
    joined = " ".join(chunk.text for chunk in chunks)
    assert all(f"token{i}" in joined for i in range(1000))
    