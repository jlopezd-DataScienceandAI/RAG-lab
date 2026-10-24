import json

import numpy as np
import pytest
from typer.testing import CliRunner

from rag.cli import _build_meta, _hash_documents, _is_up_to_date, _list_documents, app
from rag.config import Settings

HASHES = {"a.md": "h1", "b.txt": "h2"}


# --- _list_documents / _hash_documents ----------------------------------------------------------


def test_list_documents_filters_and_sorts(tmp_path):
    (tmp_path / "b.pdf").write_bytes(b"")
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "C.TXT").write_text("x")  # upper-case extension
    (tmp_path / "notes.docx").write_text("x")  # unsupported
    (tmp_path / "sub.pdf").mkdir()  # a folder with a document-like name

    assert [p.name for p in _list_documents(tmp_path)] == ["C.TXT", "a.md", "b.pdf"]


def test_list_documents_empty_folder(tmp_path):
    assert _list_documents(tmp_path) == []


def test_hash_documents_uses_file_names_as_keys(tmp_path):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.txt").write_text("hello")
    (tmp_path / "c.txt").write_text("other")
    hashes = _hash_documents(sorted(tmp_path.iterdir()))
    assert list(hashes) == ["a.md", "b.txt", "c.txt"]
    assert hashes["a.md"] == hashes["b.txt"] != hashes["c.txt"]


# --- _is_up_to_date -----------------------------------------------------------------------------


def _write_index(directory, meta):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vectors.npy").write_bytes(b"")
    (directory / "chunks.db").write_bytes(b"")
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory / "meta.json"


@pytest.fixture
def settings():
    return Settings(embedding_model="model-x", chunk_tokens=500, chunk_overlap=75)


def test_up_to_date_when_everything_matches(tmp_path, settings):
    meta_path = _write_index(tmp_path, _build_meta(HASHES, settings))
    assert _is_up_to_date(
        meta_path, dict(reversed(HASHES.items())), settings
    )  # order is irrelevant


def test_not_up_to_date_when_a_hash_changes(tmp_path, settings):
    meta_path = _write_index(tmp_path, _build_meta(HASHES, settings))
    assert not _is_up_to_date(meta_path, {**HASHES, "a.md": "changed"}, settings)


def test_not_up_to_date_when_a_document_is_added_or_removed(tmp_path, settings):
    meta_path = _write_index(tmp_path, _build_meta(HASHES, settings))
    assert not _is_up_to_date(meta_path, {**HASHES, "c.pdf": "h3"}, settings)
    assert not _is_up_to_date(meta_path, {"a.md": "h1"}, settings)


@pytest.mark.parametrize(
    "change",
    [{"embedding_model": "other"}, {"chunk_tokens": 300}, {"chunk_overlap": 50}],
)
def test_not_up_to_date_when_settings_change(tmp_path, settings, change):
    meta_path = _write_index(tmp_path, _build_meta(HASHES, settings))
    assert not _is_up_to_date(meta_path, HASHES, settings.model_copy(update=change))


@pytest.mark.parametrize("key", ["embedding_model", "chunk_tokens", "chunk_overlap", "doc_hashes"])
def test_missing_key_means_not_up_to_date(tmp_path, settings, key):
    meta = _build_meta(HASHES, settings)
    del meta[key]
    meta_path = _write_index(tmp_path, meta)
    assert not _is_up_to_date(meta_path, HASHES, settings)  # and no KeyError


def test_no_meta_file(tmp_path, settings):
    assert not _is_up_to_date(tmp_path / "meta.json", HASHES, settings)


def test_invalid_json(tmp_path, settings):
    meta_path = _write_index(tmp_path, {})
    meta_path.write_text("{not json", encoding="utf-8")
    assert not _is_up_to_date(meta_path, HASHES, settings)


def test_json_that_is_not_an_object(tmp_path, settings):
    meta_path = _write_index(tmp_path, {})
    meta_path.write_text("[1, 2]", encoding="utf-8")
    assert not _is_up_to_date(meta_path, HASHES, settings)


def test_missing_index_file_means_not_up_to_date(tmp_path, settings):
    meta_path = _write_index(tmp_path, _build_meta(HASHES, settings))
    (tmp_path / "vectors.npy").unlink()
    assert not _is_up_to_date(meta_path, HASHES, settings)


# --- the ingest command (fake embedder: no model download) --------------------------------------


class _FakeEmbedder:
    calls = 0

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode_passages(self, texts):
        type(self).calls += 1
        v = np.random.default_rng(0).normal(size=(len(texts), 8)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def project(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.md").write_text("First paragraph about credit risk.\n\nSecond paragraph.")
    (raw / "b.txt").write_text("Another document about model governance.")
    monkeypatch.chdir(tmp_path)  # so a real .env is not picked up
    monkeypatch.setenv("RAG_RAW_DIR", str(raw))
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setattr("rag.embeddings.Embedder", _FakeEmbedder)
    _FakeEmbedder.calls = 0
    return tmp_path


def test_ingest_builds_then_skips_then_rebuilds(project):
    runner = CliRunner()

    first = runner.invoke(app, ["ingest"])
    assert first.exit_code == 0, first.output
    assert "Indexed 2 documents" in first.output
    assert sorted(p.name for p in (project / "index").iterdir()) == [
        "chunks.db",
        "meta.json",
        "vectors.npy",
    ]
    meta = json.loads((project / "index" / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["doc_hashes"]) == {"a.md", "b.txt"}
    assert meta["n_chunks"] >= 2

    second = runner.invoke(app, ["ingest"])
    assert second.exit_code == 0
    assert "up to date" in second.output
    assert _FakeEmbedder.calls == 1  # the second run did not embed anything

    (project / "raw" / "a.md").write_text("The document changed.")
    third = runner.invoke(app, ["ingest"])
    assert "Indexed 2 documents" in third.output
    assert _FakeEmbedder.calls == 2


def test_ingest_explicit_folder_argument(project, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.txt").write_text("Only document.")
    result = CliRunner().invoke(app, ["ingest", str(other)])
    assert result.exit_code == 0, result.output
    assert "Indexed 1 documents" in result.output


def test_ingest_missing_folder_fails_cleanly(project):
    result = CliRunner().invoke(app, ["ingest", str(project / "nope")])
    assert result.exit_code == 1
    assert "Folder not found" in result.output


def test_ingest_folder_without_documents_fails_cleanly(project, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = CliRunner().invoke(app, ["ingest", str(empty)])
    assert result.exit_code == 1
    assert "No documents" in result.output
    assert not (project / "index").exists()  # no empty index was written
