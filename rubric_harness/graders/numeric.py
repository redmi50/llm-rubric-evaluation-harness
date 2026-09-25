"""Numeric answer grading.

Extracts a number from a free text response and compares it with the reference
value. Extraction handles thousands separators, currency symbols, a trailing
percent sign, and a negative number written in parentheses.

Scoring is deliberately coarse and stated in full so that a score can be
explained to a stakeholder:

* within tolerance            -> 1.0
* within ten times tolerance  -> 0.5
* anything else, or unparsable -> 0.0
"""

from __future__ import annotations

import re

from ..tasks import Task
from . import GRADER_NUMERIC, GraderResult

_PARENTHESISED = re.compile(r"\(\s*[$£€]?\s*([\d,]+(?:\.\d+)?)\s*\)")
_NUMBER = re.compile(r"(?P<sign>[-+])?\s*(?P<digits>\d[\d,]*(?:\.\d+)?)")
_CURRENCY = re.compile(r"[$£€]")


def extract_number(text: str) -> float | None:
    """Return the first number found in the text, or None if there is none.

    A number in parentheses is read as negative, which is the accounting
    convention. A leading minus sign is also read as negative. A percent sign is
    stripped and the value is taken at face value, so 12.5% parses as 12.5. Pass
    the reference in the same units.
    """
    if text is None:
        return None
    cleaned = _CURRENCY.sub("", str(text))

    parenthesised = _PARENTHESISED.search(cleaned)
    if parenthesised:
        return -float(parenthesised.group(1).replace(",", ""))

    match = _NUMBER.search(cleaned)
    if not match:
        return None
    try:
        value = float(match.group("digits").replace(",", ""))
    except ValueError:
        return None
    return -value if match.group("sign") == "-" else value


def parse_reference(reference: str) -> float | None:
    """Parse the reference answer, which may carry a percent sign."""
    return extract_number(reference)


class NumericGrader:
    """Grades the final numeric answer of a response."""

    def __init__(self, name: str = GRADER_NUMERIC) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """The grader key used by rubrics."""
        return self._name

    def grade(self, task: Task, response_text: str) -> GraderResult | None:
        """Score the numeric answer, or return None for non-numeric tasks."""
        if not task.is_numeric:
            return None

        reference = parse_reference(task.expected_reference)
        if reference is None:
            raise ValueError(
                f"Task {task.task_id!r} is marked numeric but its reference "
                f"{task.expected_reference!r} contains no number."
            )

        observed = extract_number(response_text)
        if observed is None:
            return GraderResult(
                grader_name=self.name,
                score=0.0,
                explanation="No numeric value could be parsed from the response.",
                details={"reference": reference, "observed": None},
            )

        difference = abs(observed - reference)
        tolerance = task.tolerance

        if difference <= tolerance:
            score = 1.0
            explanation = (
                f"Answer {observed} is within the tolerance of {tolerance} "
                f"around the reference {reference}."
            )
        elif difference <= tolerance * 10:
            score = 0.5
            explanation = (
                f"Answer {observed} misses the reference {reference} by {difference:.4g}, "
                f"which is outside the tolerance of {tolerance} but within ten times it."
            )
        else:
            score = 0.0
            explanation = (
                f"Answer {observed} misses the reference {reference} by {difference:.4g}, "
                f"far outside the tolerance of {tolerance}."
            )

        return GraderResult(
            grader_name=self.name,
            score=score,
            explanation=explanation,
            details={
                "reference": reference,
                "observed": observed,
                "absolute_error": difference,
                "tolerance": tolerance,
            },
        )
