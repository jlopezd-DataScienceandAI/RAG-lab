import numpy as np
import pytest

from rag.embeddings import Embedder

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(MODEL)


# --- laziness -------------------------------------------------------------------------------


def test_model_is_not_loaded_on_construction():
    e = Embedder(MODEL)
    assert e._model is None


def test_model_is_loaded_once_and_reused(embedder):
    embedder.encode_passages(["warm up"])
    first = embedder._model
    embedder.encode_passages(["again"])
    assert first is not None
    assert embedder._model is first


# --- encode_passages: shape, dtype, normalisation ---------------------------------------------


def test_passages_shape_dtype_and_norm(embedder):
    v = embedder.encode_passages(["first text", "second text"])
    assert v.shape == (2, DIM)
    assert v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


def test_single_passage_is_still_2d(embedder):
    v = embedder.encode_passages(["only one"])
    assert v.shape == (1, DIM)


def test_empty_list_returns_empty_matrix(embedder):
    v = embedder.encode_passages([])
    assert v.ndim == 2
    assert v.shape[0] == 0
    assert v.dtype == np.float32


def test_more_texts_than_batch_size(embedder):
    texts = [f"passage number {i}" for i in range(70)]  # batch_size is 32: 3 batches
    v = embedder.encode_passages(texts)
    assert v.shape == (70, DIM)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


def test_vectors_contain_no_nan_or_inf(embedder):
    v = embedder.encode_passages(["normal text", "", "   ", "x" * 5000])
    assert np.isfinite(v).all()


# --- encode_passages: determinism and ordering ------------------------------------------------


def test_same_text_gives_same_vector(embedder):
    a = embedder.encode_passages(["determinism check"])[0]
    b = embedder.encode_passages(["determinism check"])[0]
    assert np.allclose(a, b, atol=1e-5)


def test_order_is_preserved(embedder):
    texts = ["cats are small animals", "the stock market fell", "python is a language"]
    together = embedder.encode_passages(texts)
    alone = np.vstack([embedder.encode_passages([t]) for t in texts])
    assert np.allclose(together, alone, atol=1e-4)


def test_vector_does_not_depend_on_batch_neighbours(embedder):
    target = "the same sentence"
    a = embedder.encode_passages([target, "short"])[0]
    b = embedder.encode_passages([target, "a much longer neighbour " * 50])[0]
    assert np.allclose(a, b, atol=1e-4)


def test_duplicate_texts_give_identical_rows(embedder):
    v = embedder.encode_passages(["dup", "other", "dup"])
    assert np.allclose(v[0], v[2], atol=1e-5)


# --- encode_passages: awkward inputs ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n\n\t",
        "a",
        "12345",
        "!!!???...",
        "naïve café résumé",
        "日本語のテキスト",
        "emoji 🙂🚀",
        "line one\nline two\r\nline three",
        "quotes ' \" and {braces} and [brackets]",
        "x" * 20_000,  # far beyond the model's 512-token limit
    ],
)
def test_awkward_text_still_gives_unit_vector(embedder, text):
    v = embedder.encode_passages([text])
    assert v.shape == (1, DIM)
    assert np.isfinite(v).all()
    assert np.isclose(np.linalg.norm(v[0]), 1.0, atol=1e-4)


def test_overlong_text_is_truncated_not_an_error(embedder):
    short = embedder.encode_passages(["word " * 100])[0]
    huge = embedder.encode_passages(["word " * 5000])[0]
    assert short.shape == huge.shape == (DIM,)


def test_different_texts_give_different_vectors(embedder):
    v = embedder.encode_passages(["apples and oranges", "quantum field theory"])
    assert not np.allclose(v[0], v[1], atol=1e-3)


# --- semantics --------------------------------------------------------------------------------


def test_similar_texts_are_closer(embedder):
    q = embedder.encode_query("How do I install the package?")
    p = embedder.encode_passages(["Run pip install to set it up.", "The sky is blue."])
    assert p[0] @ q > p[1] @ q


def test_dot_product_equals_cosine(embedder):
    v = embedder.encode_passages(["alpha beta", "gamma delta"])
    cosine = v[0] @ v[1] / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))
    assert np.isclose(v[0] @ v[1], cosine, atol=1e-4)


def test_self_similarity_is_one(embedder):
    v = embedder.encode_passages(["identical"])[0]
    assert np.isclose(v @ v, 1.0, atol=1e-4)


def test_scores_are_within_cosine_range(embedder):
    v = embedder.encode_passages(["one", "two", "three", "completely unrelated 42"])
    sims = v @ v.T
    assert sims.max() <= 1.0 + 1e-4
    assert sims.min() >= -1.0 - 1e-4


# --- encode_query -----------------------------------------------------------------------------


def test_query_is_1d_float32_unit_vector(embedder):
    q = embedder.encode_query("What is the default value?")
    assert q.shape == (DIM,)
    assert q.dtype == np.float32
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-4)


@pytest.mark.parametrize("text", ["", " ", "?", "naïve ünïcode", "q " * 3000])
def test_awkward_queries(embedder, text):
    q = embedder.encode_query(text)
    assert q.shape == (DIM,)
    assert np.isfinite(q).all()


def test_query_is_deterministic(embedder):
    a = embedder.encode_query("same question")
    b = embedder.encode_query("same question")
    assert np.allclose(a, b, atol=1e-5)


def test_query_dim_matches_passage_dim(embedder):
    q = embedder.encode_query("dim check")
    p = embedder.encode_passages(["dim check"])
    assert q.shape[0] == p.shape[1]


def test_query_search_against_matrix(embedder):
    """The exact operation the store will do: matrix @ query gives one score per passage."""
    p = embedder.encode_passages(["a", "b", "c", "d"])
    q = embedder.encode_query("a")
    scores = p @ q
    assert scores.shape == (4,)


# --- query prefix (first experiment of step 8) ---------------------------------------------------


def test_prefix_changes_the_query_vector():
    with_prefix = Embedder(MODEL, use_query_prefix=True).encode_query("install the package")
    without = Embedder(MODEL, use_query_prefix=False).encode_query("install the package")
    assert not np.allclose(with_prefix, without, atol=1e-4)


def test_prefix_is_on_by_default(embedder):
    explicit = Embedder(MODEL, use_query_prefix=True).encode_query("default check")
    assert np.allclose(embedder.encode_query("default check"), explicit, atol=1e-5)


def test_prefix_does_not_affect_passages():
    a = Embedder(MODEL, use_query_prefix=True).encode_passages(["same passage"])
    b = Embedder(MODEL, use_query_prefix=False).encode_passages(["same passage"])
    assert np.allclose(a, b, atol=1e-5)


def test_query_without_prefix_equals_passage_encoding():
    e = Embedder(MODEL, use_query_prefix=False)
    q = e.encode_query("plain text")
    p = e.encode_passages(["plain text"])[0]
    assert np.allclose(q, p, atol=1e-4)
