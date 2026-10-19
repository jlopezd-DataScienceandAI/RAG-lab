"""Run the whole evaluation and write one JSON file per run into results/."""

from rag.config import Settings


def run_eval(settings: Settings) -> dict:
    """TODO(v0.1):
    1. load EvalCase rows from settings.eval_path
    2. retrieval metrics (Hit@k, MRR) for every case, no LLM involved
    3. generation: answer, judge verdict, citation check, abstention on unanswerable cases
    4. save results/<version>/<timestamp>.json with the config used and all metrics

    Change one variable at a time between runs, so you know what improved what.
    """
    raise NotImplementedError("run_eval")
