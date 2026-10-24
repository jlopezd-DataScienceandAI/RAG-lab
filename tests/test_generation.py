"""Pure functions of generation.py: no network, no model."""

from rag.generation import build_prompt, citations_are_valid, cited_ids, is_abstention
from rag.models import Chunk, RetrievalResult

ABSTENTION = "I don't know based on the provided documents."


def _result(chunk_id: str, text: str = "some text", rank: int = 1) -> RetrievalResult:
    chunk = Chunk(id=chunk_id, doc="a.pdf", text=text, n_tokens=1, doc_hash="h")
    return RetrievalResult(chunk=chunk, score=0.5, rank=rank)


RESULTS = [_result("a.pdf::p1::c000"), _result("a.pdf::p2::c001", rank=2)]


# --- build_prompt -------------------------------------------------------------------------------


def test_build_prompt_labels_each_chunk_with_its_id():
    prompt = build_prompt("{context}\nQ: {question}", "why?", RESULTS)
    assert "[a.pdf::p1::c000]\nsome text" in prompt
    assert "[a.pdf::p2::c001]\nsome text" in prompt
    assert prompt.endswith("Q: why?")


def test_build_prompt_keeps_braces_inside_chunk_text():
    results = [_result("a.pdf::p1::c000", text="config = {x} and {0}")]
    assert "config = {x} and {0}" in build_prompt("{context} {question}", "q", results)


# --- cited_ids ----------------------------------------------------------------------------------


def test_cited_ids_extracts_unique_ids():
    answer = "A [a.pdf::p1::c000]. B [a.pdf::p2::c001][a.pdf::p1::c000]."
    assert cited_ids(answer) == {"a.pdf::p1::c000", "a.pdf::p2::c001"}


def test_cited_ids_ignores_brackets_without_double_colon():
    assert cited_ids("See [1] and [note] and []") == set()


# --- citations_are_valid ------------------------------------------------------------------------


def test_citations_valid_when_all_ids_were_retrieved():
    assert citations_are_valid("Yes [a.pdf::p1::c000].", RESULTS)


def test_citations_invalid_when_an_id_was_invented():
    assert not citations_are_valid("Yes [a.pdf::p1::c000] [x.pdf::p9::c999].", RESULTS)


def test_no_citations_is_invalid_unless_abstaining():
    assert not citations_are_valid("Yes, it is.", RESULTS)
    assert citations_are_valid(ABSTENTION, RESULTS)


# --- is_abstention ------------------------------------------------------------------------------


def test_is_abstention_tolerates_case_spacing_and_curly_apostrophe():
    assert is_abstention(ABSTENTION)
    assert is_abstention("i don’t know   based on the provided\ndocuments")
    assert is_abstention('"I don\'t know based on the provided documents."')


def test_is_abstention_false_for_a_normal_answer():
    assert not is_abstention("The limit is 500 tokens [a.pdf::p1::c000].")
