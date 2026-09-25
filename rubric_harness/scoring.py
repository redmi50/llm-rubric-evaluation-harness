"""Applying a rubric to grader output.

A rubric criterion names the grader that supplies it, in its ``grader`` field.
Two kinds of name are understood:

* a plain grader key such as ``answer_accuracy`` or ``key_facts``, which resolves
  to the grader of that name, for example the numeric grader;
* a holistic key of the form ``holistic:<criterion>``, which asks the judge
  grader to grade that one criterion and returns a score keyed on the criterion
  so that two criteria handled by the same judge stay separate.

A criterion with no matching grader is dropped and the remaining weights are
renormalised, so an offline run with no judge still produces a meaningful score
instead of a silently deflated one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graders import GraderResult
from .rubric import Rubric
from .tasks import Task

MIN_LEVEL = 1
MAX_LEVEL = 5
LEVEL_SPAN = MAX_LEVEL - MIN_LEVEL
HOLISTIC_PREFIX = "holistic"


def score_to_level(score: float) -> int:
    """Map a grader score in [0, 1] onto the rubric's 1 to 5 level scale.

    The mapping is linear and rounded: 0.0 maps to level 1, 0.25 to level 2,
    0.5 to level 3, 0.75 to level 4 and 1.0 to level 5.
    """
    bounded = min(max(score, 0.0), 1.0)
    return int(round(MIN_LEVEL + bounded * LEVEL_SPAN))


def split_graders(graders) -> tuple[dict, list]:
    """Separate plain graders from judges that can grade a single criterion."""
    direct: dict = {}
    judges: list = []
    for grader in graders:
        if hasattr(grader, "grade_criterion"):
            judges.append(grader)
        else:
            direct[grader.name] = grader
    return direct, judges


def grade_response(rubric: Rubric, task: Task, response_text: str, graders) -> dict[str, GraderResult]:
    """Produce a grader result for every rubric criterion a grader can supply.

    Only the graders the rubric actually names are called, so no provider call is
    spent on a criterion the rubric does not use.
    """
    direct, judges = split_graders(graders)
    results: dict[str, GraderResult] = {}

    for criterion in rubric.criteria:
        key = criterion.grader
        if key.startswith(f"{HOLISTIC_PREFIX}:"):
            if not judges:
                continue
            result = judges[0].grade_criterion(task, response_text, criterion)
        elif key in direct:
            result = direct[key].grade(task, response_text)
        else:
            continue

        if result is not None:
            results[result.grader_name] = result

    return results


@dataclass
class CriterionScore:
    """The outcome for one rubric criterion."""

    criterion_name: str
    weight: float
    score: float
    level: int
    explanation: str
    applicable: bool = True

    def as_dict(self) -> dict:
        """Serializable representation."""
        return {
            "criterion": self.criterion_name,
            "weight": round(self.weight, 4),
            "score": round(self.score, 4),
            "level": self.level,
            "applicable": self.applicable,
            "explanation": self.explanation,
        }


@dataclass
class RubricScore:
    """The graded result for one response against one rubric."""

    task_id: str
    provider_name: str
    rubric_id: str
    criterion_scores: list[CriterionScore] = field(default_factory=list)
    weighted_total: float = 0.0
    passed: bool = False
    skipped_criteria: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def matched_weight(self) -> float:
        """The total weight that was actually applicable."""
        return sum(item.weight for item in self.criterion_scores if item.applicable)

    def as_dict(self) -> dict:
        """Serializable representation."""
        return {
            "task_id": self.task_id,
            "provider_name": self.provider_name,
            "rubric_id": self.rubric_id,
            "weighted_total": round(self.weighted_total, 4),
            "passed": self.passed,
            "skipped_criteria": list(self.skipped_criteria),
            "criteria": [item.as_dict() for item in self.criterion_scores],
            "notes": self.notes,
        }


def apply_rubric(rubric: Rubric, grader_results: dict[str, GraderResult]) -> RubricScore:
    """Score one response against a rubric using the supplied grader results."""
    criterion_scores: list[CriterionScore] = []
    applicable_weight = 0.0
    weighted_sum = 0.0
    skipped: list[str] = []

    for criterion in rubric.criteria:
        result = grader_results.get(criterion.grader)
        if result is None:
            skipped.append(criterion.name)
            criterion_scores.append(
                CriterionScore(
                    criterion_name=criterion.name,
                    weight=criterion.weight,
                    score=0.0,
                    level=MIN_LEVEL,
                    explanation=(
                        f"No grader named {criterion.grader!r} was available in this run, "
                        "so the criterion was excluded from the total."
                    ),
                    applicable=False,
                )
            )
            continue

        applicable_weight += criterion.weight
        weighted_sum += criterion.weight * result.score
        criterion_scores.append(
            CriterionScore(
                criterion_name=criterion.name,
                weight=criterion.weight,
                score=result.score,
                level=score_to_level(result.score),
                explanation=result.explanation,
                applicable=True,
            )
        )

    if applicable_weight > 0:
        weighted_total = weighted_sum / applicable_weight
        notes = ""
    else:
        weighted_total = 0.0
        notes = "No criteria could be scored because no applicable grader was supplied."

    return RubricScore(
        task_id="",
        provider_name="",
        rubric_id=rubric.rubric_id,
        criterion_scores=criterion_scores,
        weighted_total=weighted_total,
        passed=weighted_total >= rubric.pass_threshold,
        skipped_criteria=skipped,
        notes=notes,
    )


def evaluate_run(records, tasks_by_id: dict[str, Task], rubrics: dict[str, Rubric], graders) -> list[RubricScore]:
    """Grade every response in a run and return the scores in a stable order."""
    scores: list[RubricScore] = []
    for record in sorted(records, key=lambda item: (item.task_id, item.provider_name)):
        task = tasks_by_id.get(record.task_id)
        if task is None:
            raise KeyError(f"Run refers to unknown task {record.task_id!r}")

        rubric = rubrics.get(task.rubric_id)
        if rubric is None:
            raise KeyError(
                f"Task {task.task_id!r} refers to unknown rubric {task.rubric_id!r}"
            )

        grader_results = grade_response(rubric, task, record.response_text, graders)

        score = apply_rubric(rubric, grader_results)
        score.task_id = task.task_id
        score.provider_name = record.provider_name
        scores.append(score)

    return scores
