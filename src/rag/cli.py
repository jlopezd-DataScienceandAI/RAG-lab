"""Command line entry point: ``rag ingest``, ``rag ask`` and ``rag eval``."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="A small RAG pipeline built from scratch.", no_args_is_help=True)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Folder with the documents to index.")] = Path(
        "data/raw"
    ),
) -> None:
    """Read documents, chunk them, embed them and save the index."""
    typer.echo(f"TODO: ingest {path}")
    raise typer.Exit(code=1)


@app.command()
def ask(question: Annotated[str, typer.Argument(help="Your question.")]) -> None:
    """Answer a question using the saved index, with citations."""
    typer.echo(f"TODO: answer {question!r}")
    raise typer.Exit(code=1)


@app.command(name="eval")
def run_evaluation() -> None:
    """Run the evaluation dataset and write the results to results/."""
    typer.echo("TODO: run the evaluation")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
