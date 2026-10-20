from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from rag.ingest.loaders import RawPage, load_document

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_PDFS = sorted(RAW_DIR.glob("*.pdf"))


def test_markdown_is_one_page(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\n\nSome text.", encoding="utf-8")
    pages = load_document(f)
    assert len(pages) == 1
    assert pages[0].doc == "note.md"
    assert pages[0].page is None
    assert "Some text." in pages[0].text


def test_txt_is_one_page(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("plain text", encoding="utf-8")
    assert load_document(f) == [RawPage(doc="note.txt", page=None, text="plain text")]


def test_utf8_accents_are_preserved(tmp_path):
    f = tmp_path / "es.md"
    f.write_text("¿Qué protección ofrece el artículo 5?", encoding="utf-8")
    assert load_document(f)[0].text == "¿Qué protección ofrece el artículo 5?"


def test_extension_is_case_insensitive(tmp_path):
    f = tmp_path / "NOTE.MD"
    f.write_text("hello", encoding="utf-8")
    assert load_document(f)[0].text == "hello"


def test_doc_is_file_name_not_full_path(tmp_path):
    sub = tmp_path / "nested" / "folder"
    sub.mkdir(parents=True)
    f = sub / "deep.md"
    f.write_text("x", encoding="utf-8")
    assert load_document(f)[0].doc == "deep.md"


def test_windows_line_breaks_are_normalised(tmp_path):
    f = tmp_path / "crlf.md"
    f.write_bytes(b"first paragraph\r\n\r\nsecond paragraph")
    assert "\r" not in load_document(f)[0].text


def test_unsupported_extension(tmp_path):
    f = tmp_path / "data.xlsx"
    f.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(f)


def test_blank_pdf_pages_are_skipped(tmp_path):
    f = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with f.open("wb") as handle:
        writer.write(handle)
    assert load_document(f) == []


def test_utf8_bom_is_removed(tmp_path):
    f = tmp_path / "bom.md"
    f.write_bytes(b"\xef\xbb\xbf# Title")  # what Windows editors write with "UTF-8 with BOM"
    assert load_document(f)[0].text == "# Title"


def test_empty_markdown_gives_no_pages(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("  \n\n", encoding="utf-8")
    assert load_document(f) == []


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.md")


def test_pdf_extension_is_case_insensitive(tmp_path):
    f = tmp_path / "BLANK.PDF"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with f.open("wb") as handle:
        writer.write(handle)
    assert load_document(f) == []  # reached the PDF branch instead of "Unsupported"


def test_skipped_pdf_pages_keep_original_numbers(tmp_path, monkeypatch):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, _path):
            self.pages = [FakePage("one"), FakePage(""), FakePage(None), FakePage("four")]

    monkeypatch.setattr("rag.ingest.loaders.PdfReader", FakeReader)
    pages = load_document(tmp_path / "doc.pdf")
    assert [(p.page, p.text) for p in pages] == [(1, "one"), (4, "four")]


@pytest.mark.skipif(not RAW_PDFS, reason="no PDFs in data/raw")
@pytest.mark.parametrize("pdf", RAW_PDFS, ids=lambda p: p.name)
def test_real_pdfs_keep_page_numbers(pdf):
    total = len(PdfReader(pdf).pages)
    pages = load_document(pdf)
    numbers = [p.page for p in pages]
    assert pages, "no text extracted"
    assert all(p.doc == pdf.name for p in pages)
    assert all(p.text.strip() for p in pages)
    assert numbers == sorted(set(numbers))  # strictly increasing, no repeats
    assert numbers[0] >= 1 and numbers[-1] <= total  # 1-based, within the document
