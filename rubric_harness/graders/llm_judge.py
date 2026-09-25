"""Holistic grading delegated to a model acting as a judge.

The judge is shown the criterion, its level anchors, the question, the reference
answer and the response, and is asked for an integer level between 1 and 5 plus
its reasoning. Showing the anchors is the point of a rubric: the judge is
choosing between written definitions rather than applying its own taste.

The reply is parsed with a strict pattern first and a fallback pattern second. If
neither matches, the grader refuses to guess: it returns a neutral level with an
explanation that records the parse failure, so a broken judge shows up in the
report rather than silently inflating scores.
"""

from __future__ import annotations

import re

from ..rubric import Criterion
from ..tasks import Task
from . import GRADER_HOLISTIC, GraderResult

LEVEL_PATTERN = re.compile(r"LEVEL\s*[:\-]\s*([1-5])", re.IGNORECASE)
LOOSE_PATTERN = re.compile(r"\b([1-5])\s*(?:out of|/)\s*5\b", re.IGNORECASE)
NEUTRAL_LEVEL = 3

DEFAULT_CRITERION_NAME = "overall quality"
DEFAULT_ANCHORS = {
    1: "The response fails the criterion outright.",
    2: "The response partially meets the criterion with material gaps.",
    3: "The response meets the criterion at an acceptable but unremarkable level.",
    4: "The response meets the criterion well with only minor gaps.",
    5: "The response meets the criterion fully and could be used as an example.",
}

JUDGE_INSTRUCTIONS = """You are grading a model response against a single criterion of a rubric.

Criterion: {criterion_name}
What this criterion measures: {criterion_description}

Level anchors:
1: {level_1}
2: {level_2}
3: {level_3}
4: {level_4}
5: {level_5}

Question asked: {prompt}
Reference answer: {reference}
Response under review: {response}

Reply on two lines exactly:
LEVEL: <an integer from 1 to 5>
REASON: <one or two sentences of justification>
"""


def default_criterion() -> Criterion:
    """A stand-in criterion for callers that grade without a rubric."""
    return Criterion(
        name=DEFAULT_CRITERION_NAME,
        weight=1.0,
        description="The overall quality of the response.",
        levels=dict(DEFAULT_ANCHORS),
    )


class LLMJudgeGrader:
    """Grades a response by asking a provider to act as a judge.

    The same instance grades every criterion, because each call renders its own
    prompt from the criterion it is asked about. That keeps one judge, and one
    set of calibration evidence, for the whole run.
    """

    def __init__(self, provider, name: str = GRADER_HOLISTIC) -> None:
        if provider is None:
            raise ValueError("LLMJudgeGrader requires a provider to act as the judge.")
        self.provider = provider
        self._name = name
        self.parsed = 0
        self.unparsed = 0

    @property
    def name(self) -> str:
        """The grader key prefix used by rubrics, for example ``holistic``."""
        return self._name

    def build_prompt(
        self,
        task: Task,
        response_text: str,
        criterion: Criterion | None = None,
    ) -> str:
        """Render the judge prompt for one task and one criterion."""
        criterion = criterion or default_criterion()
        anchors = {
            level: criterion.levels.get(level, "") for level in (1, 2, 3, 4, 5)
        }
        return JUDGE_INSTRUCTIONS.format(
            criterion_name=criterion.name,
            criterion_description=criterion.description or "Not stated.",
            prompt=task.prompt,
            reference=task.expected_reference,
            response=response_text,
            **{f"level_{level}": anchors[level] for level in anchors},
        )

    def parse_level(self, judge_text: str) -> tuple[int | None, str]:
        """Extract the level and the stated reason from the judge's reply."""
        reason_match = re.search(r"REASON\s*[:\-]\s*(.+)", judge_text, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else judge_text.strip()

        strict = LEVEL_PATTERN.search(judge_text)
        if strict:
            return int(strict.group(1)), reason
        loose = LOOSE_PATTERN.search(judge_text)
        if loose:
            return int(loose.group(1)), reason
        return None, reason

    def _result(
        self, grader_name: str, judge_text: str, reason: str, level: int | None
    ) -> GraderResult:
        if level is None:
            self.unparsed += 1
            return GraderResult(
                grader_name=grader_name,
                score=(NEUTRAL_LEVEL - 1) / 4,
                explanation=(
                    "The judge reply contained no parsable level, so a neutral level of "
                    f"{NEUTRAL_LEVEL} was recorded. Judge reply: {reason[:200]}"
                ),
                details={"level": NEUTRAL_LEVEL, "parsed": False, "judge_reply": judge_text},
            )

        self.parsed += 1
        return GraderResult(
            grader_name=grader_name,
            score=(level - 1) / 4,
            explanation=f"Judge assigned level {level} of 5. {reason}",
            details={"level": level, "parsed": True, "judge_reply": judge_text},
        )

    def grade(self, task: Task, response_text: str) -> GraderResult | None:
        """Grade the response as a single overall quality judgement."""
        prompt = self.build_prompt(task, response_text)
        judge_text = self.provider.complete(prompt)
        level, reason = self.parse_level(judge_text)
        return self._result(self.name, judge_text, reason, level)

    def grade_criterion(
        self, task: Task, response_text: str, criterion: Criterion
    ) -> GraderResult | None:
        """Grade the response against one rubric criterion.

        The result is keyed as ``<grader name>:<criterion name>`` so that two
        criteria handled by the same judge keep separate scores.
        """
        prompt = self.build_prompt(task, response_text, criterion)
        judge_text = self.provider.complete(prompt)
        level, reason = self.parse_level(judge_text)
        return self._result(f"{self.name}:{criterion.name}", judge_text, reason, level)
