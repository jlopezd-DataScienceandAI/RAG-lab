from pathlib import Path

import pytest

import rag
from rag.config import Settings
from rag.models import EvalCase

ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "data/eval/questions.jsonl"


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
