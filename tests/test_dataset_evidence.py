"""Every gold_evidence must appear in the text pypdf extracts from its gold_doc."""

from pathlib import Path

import pytest

from rag.ingest.loaders import load_document
from rag.models import EvalCase

ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "data/eval/questions.jsonl"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not EVAL_FILE.exists(), reason="no evaluation dataset yet"),
]


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def test_gold_evidence_is_in_the_gold_doc():
    lines = EVAL_FILE.read_text(encoding="utf-8").splitlines()
    cases = [EvalCase.model_validate_json(line) for line in lines if line.strip()]
    texts: dict[str, str] = {}
    for case in cases:
        if not case.answerable:
            continue
        if case.gold_doc not in texts:
            pages = load_document(ROOT / "data/raw" / case.gold_doc)
            texts[case.gold_doc] = _normalise(" ".join(page.text for page in pages))
        assert _normalise(case.gold_evidence) in texts[case.gold_doc], (
            f"{case.id}: evidence not found in {case.gold_doc}"
        )
