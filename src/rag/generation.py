"""Build the prompt, call Claude, and check the citations."""

import re
from pathlib import Path

import anthropic

from rag.models import RetrievalResult

# Chunk ids look like "doc.pdf::p12::c041"; requiring "::" ignores other bracketed text.
CITATION_RE = re.compile(r"\[([^\[\]]+?::[^\[\]]+?)\]")
ABSTENTION_PHRASE = "i don't know based on the provided documents"


def load_prompt(path: Path) -> str:
    """Read the versioned prompt template (prompts/answer_v1.txt)."""
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, question: str, results: list[RetrievalResult]) -> str:
    """Fill the template with the chunks, each labelled with its id so the model can cite it."""
    context = "\n\n".join(f"[{r.chunk.id}]\n{r.chunk.text}" for r in results)
    return template.format(context=context, question=question)


def generate_answer(
    question: str,
    results: list[RetrievalResult],
    model: str,
    template: str,
    api_key: str,
    max_tokens: int,
) -> str:
    """Ask Claude to answer from the retrieved chunks and return the answer text."""
    prompt = build_prompt(template, question, results)
    # The key is passed explicitly: pydantic-settings reads .env but does not export it.
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def cited_ids(answer: str) -> set[str]:
    """Return the chunk ids cited in the answer as [id] markers."""
    return set(CITATION_RE.findall(answer))


def citations_are_valid(answer: str, results: list[RetrievalResult]) -> bool:
    """True if every cited id was retrieved. No citations is valid only for an abstention."""
    cited = cited_ids(answer)
    if not cited:
        return is_abstention(answer)
    return cited <= {r.chunk.id for r in results}


def is_abstention(answer: str) -> bool:
    """True if the answer contains the abstention sentence from prompts/answer_v1.txt."""
    normalised = " ".join(answer.lower().replace("’", "'").split())
    return ABSTENTION_PHRASE in normalised
