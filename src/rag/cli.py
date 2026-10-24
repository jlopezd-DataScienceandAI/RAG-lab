"""Command line entry point: ``rag ingest``, ``rag ask`` and ``rag eval``."""

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from rag.config import Settings, get_settings
from rag.ingest.loaders import file_sha256, load_document

app = typer.Typer(help="A small RAG pipeline built from scratch.", no_args_is_help=True)

SUPPORTED_SUFFIXES = (".pdf", ".md", ".txt")
INDEX_FILES = ("vectors.npy", "chunks.db", "meta.json")


def _list_documents(path: Path) -> list[Path]:
    """Return the supported documents directly inside ``path``, sorted by name.

    Only the first level is read (no subfolders). A fixed order makes the index come out the
    same on every run.
    """
    # Sorted by name as a plain string: sorting Path objects compares case-insensitively on
    # Windows and case-sensitively elsewhere, which would change the chunk order per system.
    return sorted(
        (f for f in path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda f: f.name,
    )


def _hash_documents(files: list[Path]) -> dict[str, str]:
    """Map each file name to the SHA-256 of its content (the same name Chunk.doc will carry)."""
    return {f.name: file_sha256(f) for f in files}


def _build_meta(hashes: dict[str, str], settings: Settings) -> dict:
    """What describes how an index was built: saved as extra_meta and compared on the next run.

    Using one function for both directions keeps the keys identical in both places.
    """
    return {
        "embedding_model": settings.embedding_model,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap": settings.chunk_overlap,
        "doc_hashes": hashes,
    }


def _is_up_to_date(meta_path: Path, hashes: dict[str, str], settings: Settings) -> bool:
    """True if the saved index was built from exactly these documents and settings.

    Anything unexpected (no index, unreadable meta.json, a missing key) means "not up to date".
    """
    if not all((meta_path.parent / name).is_file() for name in INDEX_FILES):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError covers invalid JSON and bad encoding
        return False
    if not isinstance(meta, dict):
        return False
    expected = _build_meta(hashes, settings)
    return all(meta.get(key) == value for key, value in expected.items())


def _fail(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code=1)


@app.command()
def ingest(
    path: Annotated[
        Path | None,
        typer.Argument(help="Folder with the documents to index (default: RAG_RAW_DIR)."),
    ] = None,
) -> None:
    """Read documents, chunk them, embed them and save the index."""
    settings = get_settings()
    folder = path or settings.raw_dir
    if not folder.is_dir():
        raise _fail(f"Folder not found: {folder}")

    files = _list_documents(folder)
    if not files:
        raise _fail(f"No documents ({', '.join(SUPPORTED_SUFFIXES)}) found in {folder}")

    hashes = _hash_documents(files)
    if _is_up_to_date(settings.index_dir / "meta.json", hashes, settings):
        typer.echo("Index is up to date, nothing to do.")
        return

    # Imported here so the helpers above can be used (and tested) without loading tiktoken,
    # numpy or the embedding model.
    from rag.embeddings import Embedder
    from rag.ingest.chunking import chunk_pages
    from rag.store.numpy_store import NumpyStore

    start = time.perf_counter()

    chunks = []
    for file in files:
        pages = load_document(file)
        chunks.extend(
            chunk_pages(pages, settings.chunk_tokens, settings.chunk_overlap, hashes[file.name])
        )
    if not chunks:
        raise _fail("The documents contain no extractable text (scanned PDFs?).")

    embedder = Embedder(settings.embedding_model)
    vectors = embedder.encode_passages([chunk.text for chunk in chunks])  # one call for all

    store = NumpyStore()
    store.add(chunks, vectors)
    store.save(settings.index_dir, extra_meta=_build_meta(hashes, settings))

    seconds = time.perf_counter() - start
    typer.echo(f"Indexed {len(files)} documents, {len(chunks)} chunks in {seconds:.1f} s.")


@app.command()
def ask(question: Annotated[str, typer.Argument(help="Your question.")]) -> None:
    """Answer a question using the saved index, with citations."""
    if not question.strip():
        raise _fail("The question is empty.")
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise _fail("ANTHROPIC_API_KEY is not set: add it to your .env file.")
    if not all((settings.index_dir / name).is_file() for name in INDEX_FILES):
        raise _fail(f"No index in {settings.index_dir}: run `rag ingest` first.")

    # Imported here to keep `rag --help` fast (torch takes seconds to import).
    from rag.embeddings import Embedder
    from rag.generation import cited_ids, generate_answer, load_prompt
    from rag.retrieval import retrieve
    from rag.store.numpy_store import NumpyStore

    store = NumpyStore()
    store.load(settings.index_dir)
    embedder = Embedder(settings.embedding_model)

    results = retrieve(question, embedder, store, settings.top_k)
    answer = generate_answer(
        question,
        results,
        settings.generation_model,
        load_prompt(settings.prompt_path),
        settings.anthropic_api_key,
        settings.max_answer_tokens,
    )

    typer.echo(answer)
    cited = cited_ids(answer)
    sources = [r for r in results if r.chunk.id in cited]
    if sources:
        typer.echo("\nSources:")
        for r in sources:
            preview = " ".join(r.chunk.text.split())[:100]
            typer.echo(f"  [{r.chunk.id}] {preview}")


@app.command(name="eval")
def run_evaluation() -> None:
    """Run the evaluation dataset and write the results to results/."""
    typer.echo("TODO: run the evaluation")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
