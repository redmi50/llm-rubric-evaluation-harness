"""Graders and their shared result type.

A grader inspects one response for one task and returns a score between 0 and 1
with a written explanation. Graders return ``None`` when they do not apply to a
task, for example a numeric grader applied to a written recommendation, and the
rubric then drops the criteria that depended on them.

The shared constants and the result type are declared before the grader modules
are imported, because each grader module imports them back from this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Grader keys referenced by the ``grader`` field on rubric criteria.
GRADER_NUMERIC = "answer_accuracy"
GRADER_KEYWORD = "key_facts"
GRADER_HOLISTIC = "holistic"


@dataclass
class GraderResult:
    """The outcome of grading one response."""

    grader_name: str
    score: float
    explanation: str
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"Grader {self.grader_name!r} produced {self.score}, "
                "but scores must be between 0 and 1."
            )


@runtime_checkable
class Grader(Protocol):
    """Anything that can score a response against a task."""

    @property
    def name(self) -> str:
        """The grader key used by rubrics."""

    def grade(self, task, response_text: str) -> "GraderResult | None":
        """Score the response, or return None when the grader does not apply."""


from .keyword import KeywordGrader  # noqa: E402
from .llm_judge import LLMJudgeGrader  # noqa: E402
from .numeric import NumericGrader  # noqa: E402


def default_graders(judge_provider=None) -> list:
    """The grader set used by the command line interface.

    The holistic grader is only included when a judge provider is supplied, so
    the offline default run grades on answer accuracy and key fact coverage.
    """
    graders: list = [NumericGrader(), KeywordGrader()]
    if judge_provider is not None:
        graders.append(LLMJudgeGrader(judge_provider))
    return graders


def grader_names() -> tuple[str, ...]:
    """The grader keys the default offline run can supply."""
    return (GRADER_NUMERIC, GRADER_KEYWORD, GRADER_HOLISTIC)


__all__ = [
    "Grader",
    "GraderResult",
    "KeywordGrader",
    "LLMJudgeGrader",
    "NumericGrader",
    "default_graders",
    "grader_names",
    "GRADER_NUMERIC",
    "GRADER_KEYWORD",
    "GRADER_HOLISTIC",
]
