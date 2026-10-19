"""Build the prompt, call Claude, and check the citations."""

from pathlib import Path

from rag.models import RetrievalResult


def load_prompt(path: Path) -> str:
    """Read the versioned prompt template (prompts/answer_v1.txt)."""
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, question: str, results: list[RetrievalResult]) -> str:
    """TODO(v0.1): fill {context} with the chunks (each labelled with its id) and {question}."""
    raise NotImplementedError("build_prompt")


def generate_answer(
    question: str, results: list[RetrievalResult], model: str, template: str
) -> str:
    """TODO(v0.1): call the Anthropic SDK (messages.create) and return the answer text."""
    raise NotImplementedError("generate_answer")


def cited_ids(answer: str) -> set[str]:
    """TODO(v0.1): extract the chunk ids the answer cites, e.g. with a regex on [id] markers."""
    raise NotImplementedError("cited_ids")


def citations_are_valid(answer: str, results: list[RetrievalResult]) -> bool:
    """TODO(v0.1): True if every cited id is among the retrieved chunks (programmatic check)."""
    raise NotImplementedError("citations_are_valid")
