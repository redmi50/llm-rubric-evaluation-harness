"""LLM Rubric Evaluation Harness.

Submits a fixed task suite to several language model providers, grades every
response against a weighted rubric, measures inter-grader agreement for
calibration, and exports structured written feedback reports.

Runs fully offline by default through a deterministic mock provider, so the
repository can be cloned and executed with no API key.

Public API:
    from rubric_harness import Rubric, load_tasks, run_matrix, build_report
"""

from .rubric import Criterion, Rubric, RubricValidationError, load_rubrics
from .tasks import Task, load_tasks
from .runner import ResponseRecord, run_matrix
from .scoring import RubricScore, apply_rubric, evaluate_run, grade_response
from .graders import GraderResult, default_graders
from .providers import build_provider, get_provider
from .report import build_report, to_markdown
from .agreement import cohen_kappa, percent_agreement

__all__ = [
    "Criterion",
    "Rubric",
    "RubricValidationError",
    "load_rubrics",
    "Task",
    "load_tasks",
    "ResponseRecord",
    "run_matrix",
    "RubricScore",
    "apply_rubric",
    "evaluate_run",
    "grade_response",
    "GraderResult",
    "default_graders",
    "get_provider",
    "build_provider",
    "build_report",
    "to_markdown",
    "cohen_kappa",
    "percent_agreement",
]

__version__ = "1.0.0"
