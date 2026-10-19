"""LLM-as-judge for answer quality, with a fixed rubric that you write and version."""

from pydantic import BaseModel


class JudgeVerdict(BaseModel):
    faithful: bool  # every claim is supported by the retrieved context
    correct: bool  # matches the expected answer
    abstained: bool  # the system said it does not know
    notes: str = ""


def judge_answer(
    question: str, answer: str, context: str, expected: str | None, model: str
) -> JudgeVerdict:
    """TODO(v0.1): ask Claude to grade the answer against a fixed rubric and parse the verdict.

    Tip: ask for JSON only, validate it with JudgeVerdict, and keep the rubric in prompts/.
    """
    raise NotImplementedError("judge_answer")
