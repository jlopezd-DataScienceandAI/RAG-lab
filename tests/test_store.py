import json
import sqlite3

import numpy as np
import pytest

from rag.models import Chunk
from rag.store.base import VectorStore
from rag.store.numpy_store import NumpyStore


def _chunks(n: int, start: int = 0) -> list[Chunk]:
    return [
        Chunk(
            id=f"a.md::p0::c{i:03d}",
            doc="a.md",
            page=None,
            section=None,
            text=f"text {i}",
            n_tokens=3,
            doc_hash="h",
        )
        for i in range(start, start + n)
    ]


def _vectors(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _filled(n: int = 20, dim: int = 8) -> tuple[NumpyStore, np.ndarray]:
    store = NumpyStore()
    v = _vectors(n, dim)
    store.add(_chunks(n), v)
    return store, v


# --- interface --------------------------------------------------------------------------------


def test_interface_cannot_be_instantiated():
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


def test_numpy_store_starts_empty():
    assert len(NumpyStore()) == 0


# --- add --------------------------------------------------------------------------------------


def test_add_updates_length():
    store, _ = _filled(7)
    assert len(store) == 7


def test_add_mismatched_lengths_raises():
    store = NumpyStore()
    with pytest.raises(ValueError):
        store.add(_chunks(3), _vectors(4))


def test_add_mismatch_leaves_store_untouched():
    store, _ = _filled(5)
    with pytest.raises(ValueError):
        store.add(_chunks(3, start=5), _vectors(2))
    assert len(store) == 5


def test_add_in_two_batches_appends():
    store = NumpyStore()
    v = _vectors(10)
    store.add(_chunks(6), v[:6])
    store.add(_chunks(4, start=6), v[6:])
    assert len(store) == 10
    assert store.search(v[8], k=1)[0].chunk.id == "a.md::p0::c008"


def test_add_wrong_dimension_on_second_batch_raises():
    store = NumpyStore()
    store.add(_chunks(3), _vectors(3, dim=8))
    with pytest.raises(ValueError):
        store.add(_chunks(2, start=3), _vectors(2, dim=16))
    assert len(store) == 3


def test_add_empty_batch_is_a_noop():
    store, _ = _filled(4)
    store.add([], np.empty((0, 8), dtype=np.float32))
    assert len(store) == 4


def test_add_empty_batch_on_empty_store():
    store = NumpyStore()
    store.add([], np.empty((0, 8), dtype=np.float32))
    assert len(store) == 0
    assert store.search(_vectors(1)[0], k=3) == []


def test_add_converts_float64_to_float32():
    store = NumpyStore()
    v = _vectors(3).astype(np.float64)
    store.add(_chunks(3), v)
    assert store._vectors is not None
    assert store._vectors.dtype == np.float32


def test_add_does_not_alias_the_caller_array():
    store = NumpyStore()
    v = _vectors(3)
    store.add(_chunks(3), v)
    expected_first = store.search(v[0], k=1)[0].chunk.id
    v[:] = 0.0  # caller mutates its own array afterwards
    assert store.search(_vectors(3)[0], k=1)[0].chunk.id == expected_first


def test_add_rejects_one_dimensional_vectors():
    store = NumpyStore()
    with pytest.raises(ValueError):
        store.add(_chunks(1), _vectors(1)[0])


# --- search -----------------------------------------------------------------------------------


def test_own_vector_ranks_first():
    store, v = _filled(50)
    results = store.search(v[17], k=5)
    assert results[0].chunk.id == "a.md::p0::c017"
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]


def test_scores_are_sorted_descending():
    store, v = _filled(60)
    scores = [r.score for r in store.search(v[0], k=10)]
    assert scores == sorted(scores, reverse=True)


def test_top_score_of_own_vector_is_about_one():
    store, v = _filled(10)
    assert store.search(v[4], k=1)[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_plain_python_floats():
    store, v = _filled(5)
    assert all(type(r.score) is float for r in store.search(v[0], k=3))


def test_search_matches_brute_force_ranking():
    store, v = _filled(40)
    q = _vectors(1, seed=99)[0]
    expected = np.argsort(-(v @ q))[:7]
    got = [int(r.chunk.id[-3:]) for r in store.search(q, k=7)]
    assert got == expected.tolist()


def test_k_larger_than_store_returns_everything():
    store, v = _filled(4)
    results = store.search(v[0], k=100)
    assert len(results) == 4
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert len({r.chunk.id for r in results}) == 4


def test_k_equal_to_store_size():
    store, v = _filled(6)
    assert len(store.search(v[0], k=6)) == 6


def test_k_one():
    store, v = _filled(6)
    results = store.search(v[2], k=1)
    assert len(results) == 1
    assert results[0].rank == 1


def test_search_on_empty_store_returns_empty_list():
    assert NumpyStore().search(_vectors(1)[0], k=5) == []


def test_single_chunk_store():
    store, v = _filled(1)
    results = store.search(v[0], k=5)
    assert len(results) == 1
    assert results[0].chunk.id == "a.md::p0::c000"


def test_k_zero_returns_empty_or_raises_cleanly():
    store, v = _filled(5)
    try:
        assert store.search(v[0], k=0) == []
    except ValueError:
        pass  # rejecting k < 1 is also acceptable


def test_negative_k_does_not_return_results():
    store, v = _filled(5)
    try:
        assert store.search(v[0], k=-1) == []
    except ValueError:
        pass


def test_identical_vectors_do_not_crash_and_all_come_back():
    store = NumpyStore()
    one = _vectors(1)
    store.add(_chunks(5), np.repeat(one, 5, axis=0))
    results = store.search(one[0], k=5)
    assert len(results) == 5
    assert len({r.chunk.id for r in results}) == 5


def test_search_accepts_float64_query():
    store, v = _filled(10)
    assert store.search(v[3].astype(np.float64), k=1)[0].chunk.id == "a.md::p0::c003"


def test_search_does_not_modify_the_query():
    store, v = _filled(10)
    q = v[3].copy()
    store.search(q, k=3)
    assert np.array_equal(q, v[3])


def test_search_wrong_query_dimension_raises():
    store, _ = _filled(5, dim=8)
    with pytest.raises(ValueError):
        store.search(_vectors(1, dim=16)[0], k=2)


def test_chunk_metadata_survives_in_results():
    store = NumpyStore()
    chunk = Chunk(
        id="g.pdf::p12::c041",
        doc="g.pdf",
        page=12,
        section="Intro",
        text="hello",
        n_tokens=1,
        doc_hash="abc",
    )
    v = _vectors(1)
    store.add([chunk], v)
    assert store.search(v[0], k=1)[0].chunk == chunk


# --- save / load ------------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    store, v = _filled(20)
    store.save(tmp_path)
    loaded = NumpyStore()
    loaded.load(tmp_path)
    assert len(loaded) == 20
    assert loaded.search(v[3], k=1)[0].chunk.id == "a.md::p0::c003"
    assert not list(tmp_path.glob("*.tmp.*"))  # no temporary files left behind


def test_save_creates_the_three_files(tmp_path):
    store, _ = _filled(5)
    store.save(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"vectors.npy", "chunks.db", "meta.json"}


def test_save_creates_missing_directories(tmp_path):
    store, _ = _filled(3)
    target = tmp_path / "deep" / "nested" / "index"
    store.save(target)
    assert (target / "meta.json").exists()


def test_round_trip_preserves_every_field(tmp_path):
    chunks = [
        Chunk(
            id="g.pdf::p1::c000",
            doc="g.pdf",
            page=1,
            section="A",
            text="first",
            n_tokens=1,
            doc_hash="h1",
        ),
        Chunk(
            id="n.md::p0::c001",
            doc="n.md",
            page=None,
            section=None,
            text="second",
            n_tokens=2,
            doc_hash="h2",
        ),
    ]
    store = NumpyStore()
    v = _vectors(2)
    store.add(chunks, v)
    store.save(tmp_path)
    loaded = NumpyStore()
    loaded.load(tmp_path)
    assert [r.chunk for r in loaded.search(v[0], k=2)] == [r.chunk for r in store.search(v[0], k=2)]


def test_round_trip_preserves_vectors_exactly(tmp_path):
    store, v = _filled(15)
    store.save(tmp_path)
    loaded = NumpyStore()
    loaded.load(tmp_path)
    assert loaded._vectors is not None
    assert loaded._vectors.dtype == np.float32
    assert np.array_equal(loaded._vectors, v)


def test_row_idx_aligns_with_vector_rows(tmp_path):
    store, _ = _filled(12)
    store.save(tmp_path)
    con = sqlite3.connect(tmp_path / "chunks.db")
    try:
        rows = con.execute("SELECT row_idx, id FROM chunks ORDER BY row_idx").fetchall()
    finally:
        con.close()
    assert [r[0] for r in rows] == list(range(12))
    assert [r[1] for r in rows] == [f"a.md::p0::c{i:03d}" for i in range(12)]


def test_load_preserves_order_even_if_ids_sort_differently(tmp_path):
    store = NumpyStore()
    ids = ["z.md::p0::c000", "a.md::p0::c000", "m.md::p0::c000"]
    chunks = [Chunk(id=i, doc=i.split("::")[0], text="t", n_tokens=1, doc_hash="h") for i in ids]
    v = _vectors(3)
    store.add(chunks, v)
    store.save(tmp_path)
    loaded = NumpyStore()
    loaded.load(tmp_path)
    for row, expected in enumerate(ids):
        assert loaded.search(v[row], k=1)[0].chunk.id == expected


def test_text_with_quotes_newlines_and_unicode_survives(tmp_path):
    nasty = 'It\'s a "quote"; DROP TABLE chunks;--\nline2\r\n日本語 🙂 {braces} %s ?'
    chunk = Chunk(id="x.md::p0::c000", doc="x.md", text=nasty, n_tokens=5, doc_hash="h")
    store = NumpyStore()
    v = _vectors(1)
    store.add([chunk], v)
    store.save(tmp_path)
    loaded = NumpyStore()
    loaded.load(tmp_path)
    assert loaded.search(v[0], k=1)[0].chunk.text == nasty


def test_save_empty_store_round_trips(tmp_path):
    NumpyStore().save(tmp_path)
    loaded = NumpyStore()
    loaded.load(tmp_path)
    assert len(loaded) == 0
    assert loaded.search(_vectors(1)[0], k=3) == []


def test_save_twice_overwrites_cleanly(tmp_path):
    first, _ = _filled(5)
    first.save(tmp_path)
    second, v = _filled(9)
    second.save(tmp_path)
    loaded = NumpyStore()
    loaded.load(tmp_path)
    assert len(loaded) == 9
    assert not list(tmp_path.glob("*.tmp.*"))


def test_save_after_a_stale_tmp_db_exists(tmp_path):
    (tmp_path / "chunks.tmp.db").write_bytes(b"leftover from an interrupted run")
    store, _ = _filled(4)
    store.save(tmp_path)
    loaded = NumpyStore()
    loaded.load(tmp_path)
    assert len(loaded) == 4
    assert not list(tmp_path.glob("*.tmp.*"))


def test_save_does_not_change_the_in_memory_store(tmp_path):
    store, v = _filled(6)
    store.save(tmp_path)
    assert len(store) == 6
    assert store.search(v[2], k=1)[0].chunk.id == "a.md::p0::c002"


def test_extra_meta_is_written_to_meta_json(tmp_path):
    store, _ = _filled(5, dim=8)
    store.save(tmp_path, extra_meta={"embedding_model": "m", "doc_hashes": {"a.md": "h"}})
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_chunks"] == 5
    assert meta["dim"] == 8
    assert "created_at" in meta
    assert meta["embedding_model"] == "m"
    assert meta["doc_hashes"] == {"a.md": "h"}


def test_save_without_extra_meta_still_writes_core_fields(tmp_path):
    store, _ = _filled(3)
    store.save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_chunks"] == 3


def test_load_missing_directory_raises(tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError)):
        NumpyStore().load(tmp_path / "does_not_exist")


def test_load_with_a_missing_file_raises(tmp_path):
    store, _ = _filled(3)
    store.save(tmp_path)
    (tmp_path / "vectors.npy").unlink()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        NumpyStore().load(tmp_path)


def test_load_replaces_existing_content(tmp_path):
    saved, _ = _filled(3)
    saved.save(tmp_path)
    target, _ = _filled(10)
    target.load(tmp_path)
    assert len(target) == 3


# --- L5: out-of-sync detection ------------------------------------------------------------------


def test_load_detects_wrong_n_chunks_in_meta(tmp_path):
    store, _ = _filled(10)
    store.save(tmp_path)
    meta_path = tmp_path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["n_chunks"] = 11
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="out of sync"):
        NumpyStore().load(tmp_path)


def test_load_detects_vectors_without_matching_chunks(tmp_path):
    store, _ = _filled(10)
    store.save(tmp_path)
    np.save(tmp_path / "vectors.npy", _vectors(7))  # vectors.npy replaced by another run
    with pytest.raises(RuntimeError, match="out of sync"):
        NumpyStore().load(tmp_path)


def test_load_detects_chunks_db_with_missing_rows(tmp_path):
    store, _ = _filled(10)
    store.save(tmp_path)
    con = sqlite3.connect(tmp_path / "chunks.db")
    try:
        con.execute("DELETE FROM chunks WHERE row_idx >= 8")
        con.commit()
    finally:
        con.close()
    with pytest.raises(RuntimeError, match="out of sync"):
        NumpyStore().load(tmp_path)
