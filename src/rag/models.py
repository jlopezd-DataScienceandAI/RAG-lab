"""Data models shared across the pipeline."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    """A piece of a document. Its id is human readable, e.g. ``guide.pdf::p12::c03``."""

    id: str
    doc: str
    page: int | None = None
    section: str | None = None
    text: str
    n_tokens: int
    doc_hash: str


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int = Field(ge=1)


class EvalCase(BaseModel):
    """One line of ``data/eval/questions.jsonl``."""

    id: str
    question: str
    answerable: bool = True
    gold_doc: str | None = None
    expected_answer: str | None = None
    gold_evidence: str | None = None
    tags: list[str] = Field(default_factory=list)  # e.g. ["exact_term"], ["multi_hop"]

    @model_validator(mode="after")
    def _answerable_needs_gold(self) -> Self:
        if self.answerable and not self.gold_doc:
            raise ValueError(f"{self.id}: answerable questions need a gold_doc")
        return self
