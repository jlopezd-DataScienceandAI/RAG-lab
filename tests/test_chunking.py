"""Executable spec for chunking: the invariants a chunker must keep."""

import pytest

from rag.ingest.chunking import chunk_pages, count_tokens
from rag.ingest.loaders import RawPage


def _page(text: str, doc: str = "a.md", page: int | None = None) -> RawPage:
    return RawPage(doc=doc, page=page, text=text)


def _numbered_paragraphs(n: int, min_tokens: int) -> list[str]:
    """Paragraphs of about ``min_tokens`` tokens made of globally unique, ordered words."""
    counter = 0
    paragraphs = []
    for _ in range(n):
        words: list[str] = []
        while count_tokens(" ".join(words)) < min_tokens:
            words.append(f"w{counter}")
            counter += 1
        paragraphs.append(" ".join(words))
    return paragraphs


def test_chunks_respect_max_tokens():
    pages = [_page("word " * 2000)]
    chunks = chunk_pages(pages, max_tokens=100, overlap=10)
    assert chunks
    assert all(chunk.n_tokens <= 100 for chunk in chunks)


def test_chunk_ids_are_unique_and_readable():
    pages = [_page("Some text. " * 500)]
    chunks = chunk_pages(pages, max_tokens=100, overlap=10)
    ids = [chunk.id for chunk in chunks]
    assert len(set(ids)) == len(ids)
    assert all(chunk_id.startswith("a.md::") for chunk_id in ids)


def test_no_text_is_lost():
    text = " ".join(f"token{i}" for i in range(1000))
    chunks = chunk_pages([_page(text)], max_tokens=100, overlap=10)
    joined = " ".join(chunk.text for chunk in chunks)
    assert all(f"token{i}" in joined for i in range(1000))


def test_n_tokens_matches_the_text():
    chunks = chunk_pages([_page("word " * 500)], max_tokens=100, overlap=10)
    assert all(chunk.n_tokens == count_tokens(chunk.text) for chunk in chunks)


def test_ids_continue_across_pages_and_use_page_numbers():
    pages = [_page("First page.", "g.pdf", 1), _page("Second page.", "g.pdf", 2)]
    chunks = chunk_pages(pages, max_tokens=100, overlap=10)
    assert [c.id for c in chunks] == ["g.pdf::p1::c000", "g.pdf::p2::c001"]
    assert [c.page for c in chunks] == [1, 2]


def test_page_none_is_p0_in_the_id():
    chunks = chunk_pages([_page("Some text.")], max_tokens=100, overlap=10)
    assert chunks[0].id == "a.md::p0::c000"
    assert chunks[0].page is None


def test_doc_hash_is_copied_to_every_chunk():
    chunks = chunk_pages([_page("word " * 500)], max_tokens=100, overlap=10, doc_hash="abc")
    assert chunks
    assert {c.doc_hash for c in chunks} == {"abc"}


def test_empty_input_gives_no_chunks():
    assert chunk_pages([], max_tokens=100, overlap=10) == []
    assert chunk_pages([_page("  \n\n  ")], max_tokens=100, overlap=10) == []


def test_overlap_must_be_smaller_than_max_tokens():
    with pytest.raises(ValueError):
        chunk_pages([_page("text")], max_tokens=50, overlap=50)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages([_page("text")], max_tokens=50, overlap=-1)


def test_short_paragraphs_are_packed_into_one_chunk():
    text = "First paragraph is here.\n\nSecond paragraph is here."
    chunks = chunk_pages([_page(text)], max_tokens=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_short_heading_is_glued_to_the_next_paragraph():
    body = " ".join(["Body sentence number one is a full sentence."] * 3)
    chunks = chunk_pages([_page(f"Chapter 1\n\n{body}")], max_tokens=200, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Chapter 1")


def test_heading_is_not_left_alone_in_its_own_chunk():
    paragraphs = _numbered_paragraphs(6, min_tokens=60)
    text = "\n\n".join(["Heading"] + paragraphs[:3] + ["Other heading"] + paragraphs[3:])
    chunks = chunk_pages([_page(text)], max_tokens=100, overlap=10)
    assert all(count_tokens(c.text) > 5 for c in chunks)
    assert not any(c.text in ("Heading", "Other heading") for c in chunks)


def test_consecutive_chunks_share_the_overlap():
    text = " ".join(f"w{i}" for i in range(600))
    chunks = chunk_pages([_page(text)], max_tokens=100, overlap=20)
    assert len(chunks) > 2
    for before, after in zip(chunks, chunks[1:], strict=False):
        last_words = before.text.split()[-3:]
        assert all(word in after.text for word in last_words)


def test_overlap_tail_is_contiguous_with_the_previous_chunk():
    # Paragraphs of ~70 tokens: two never fit together, and tail + paragraph exceeds the
    # budget, so the tail has to be shortened. It must keep the END of the previous chunk.
    paragraphs = _numbered_paragraphs(5, min_tokens=70)
    chunks = chunk_pages([_page("\n\n".join(paragraphs))], max_tokens=100, overlap=40)
    assert len(chunks) >= 4
    for before, after in zip(chunks, chunks[1:], strict=False):
        tail = after.text.split("\n\n")[0]
        assert tail.split()[-1] == before.text.split()[-1]


def test_long_paragraph_is_cut_with_overlap_and_loses_nothing():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_pages([_page(text)], max_tokens=60, overlap=10)
    joined = " ".join(c.text for c in chunks)
    assert all(f"w{i}" in joined for i in range(400))
    assert all(c.n_tokens <= 60 for c in chunks)


def test_cutting_inside_multibyte_characters_leaves_no_replacement_char():
    text = "Résumé 😀 日本語のテキスト “quoted” – € § " * 200
    chunks = chunk_pages([_page(text)], max_tokens=50, overlap=10)
    assert chunks
    assert not any("�" in c.text for c in chunks)
