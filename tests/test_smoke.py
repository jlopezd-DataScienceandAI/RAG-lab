from collections import Counter
from pathlib import Path

import pytest

import rag
from rag.config import Settings
from rag.models import EvalCase

ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "data/eval/questions.jsonl"
TAG_RANGES = {
    "factual": (11, 15),
    "exact_term": (4, 6),
    "multi_hop": (2, 4),
    "unanswerable": (3, 5),
}


def test_package_has_version():
    assert rag.__version__


def test_default_settings():
    settings = Settings(_env_file=None)
    assert settings.chunk_overlap < settings.chunk_tokens
    assert settings.top_k > 0


@pytest.mark.skipif(not EVAL_FILE.exists(), reason="no evaluation dataset yet")
def test_eval_dataset_parses():
    lines = EVAL_FILE.read_text(encoding="utf-8").splitlines()
    cases = [EvalCase.model_validate_json(line) for line in lines if line.strip()]
    assert cases
    assert len({case.id for case in cases}) == len(cases)


@pytest.fixture
def cases() -> list[EvalCase]:
    lines = EVAL_FILE.read_text(encoding="utf-8").splitlines()
    return [EvalCase.model_validate_json(line) for line in lines if line.strip()]


@pytest.mark.skipif(not EVAL_FILE.exists(), reason="no evaluation dataset yet")
def test_eval_dataset_follows_the_format(cases):
    raw = ROOT / "data/raw"
    for case in cases:
        assert case.tags and set(case.tags) <= set(TAG_RANGES), case.id
        if case.answerable:
            assert case.gold_evidence and case.expected_answer, case.id
            assert (raw / case.gold_doc).is_file(), f"{case.id}: {case.gold_doc} not in data/raw"
        else:
            assert case.tags == ["unanswerable"], case.id
            assert case.gold_doc is None, case.id
            assert case.gold_evidence is None, case.id
            assert case.expected_answer is None, case.id


@pytest.mark.skipif(not EVAL_FILE.exists(), reason="no evaluation dataset yet")
def test_eval_dataset_type_counts(cases):
    counts = Counter(tag for case in cases for tag in case.tags)
    assert 20 <= len(cases) <= 30
    for tag, (low, high) in TAG_RANGES.items():
        assert low <= counts[tag] <= high, f"{tag}: {counts[tag]} not in {low}-{high}"
