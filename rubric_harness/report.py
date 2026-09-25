"""Report building and rendering.

The strengths and weaknesses sections are derived from the actual per-criterion
scores rather than written by hand, so the narrative in the report cannot drift
away from the numbers in the table above it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STRENGTH_THRESHOLD = 0.8
WEAKNESS_THRESHOLD = 0.4


@dataclass
class ModelScorecard:
    """Aggregated performance for one provider."""

    provider_name: str
    mean_weighted: float
    pass_rate: float
    tasks_scored: int
    criterion_means: dict[str, float] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    skipped_criteria: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Serializable representation."""
        return {
            "provider_name": self.provider_name,
            "mean_weighted": round(self.mean_weighted, 4),
            "pass_rate": round(self.pass_rate, 4),
            "tasks_scored": self.tasks_scored,
            "criterion_means": {
                key: round(value, 4) for key, value in sorted(self.criterion_means.items())
            },
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "skipped_criteria": list(self.skipped_criteria),
        }


@dataclass
class EvaluationReport:
    """The full result of a run."""

    scorecards: list[ModelScorecard] = field(default_factory=list)
    per_task: dict[str, dict[str, float]] = field(default_factory=dict)
    best_by_task: dict[str, str] = field(default_factory=dict)
    task_count: int = 0

    @property
    def provider_names(self) -> list[str]:
        """Every provider in the report, in scorecard order."""
        return [scorecard.provider_name for scorecard in self.scorecards]

    def as_dict(self) -> dict:
        """Serializable representation."""
        return {
            "task_count": self.task_count,
            "scorecards": [scorecard.as_dict() for scorecard in self.scorecards],
            "best_by_task": dict(sorted(self.best_by_task.items())),
            "per_task": {key: dict(sorted(value.items())) for key, value in sorted(self.per_task.items())},
        }


def build_report(scores) -> EvaluationReport:
    """Aggregate rubric scores into a report."""
    by_provider: dict[str, list] = {}
    for score in scores:
        by_provider.setdefault(score.provider_name, []).append(score)

    scorecards: list[ModelScorecard] = []
    for provider_name in sorted(by_provider):
        provider_scores = by_provider[provider_name]
        totals = [score.weighted_total for score in provider_scores]
        passes = [score for score in provider_scores if score.passed]

        criterion_values: dict[str, list[float]] = {}
        skipped: list[str] = []
        for score in provider_scores:
            for item in score.criterion_scores:
                if item.applicable:
                    criterion_values.setdefault(item.criterion_name, []).append(item.score)
                elif item.criterion_name not in skipped:
                    skipped.append(item.criterion_name)

        criterion_means = {
            name: sum(values) / len(values)
            for name, values in criterion_values.items()
            if values
        }

        strengths = [
            f"{name} (mean {value:.2f})"
            for name, value in sorted(criterion_means.items())
            if value >= STRENGTH_THRESHOLD
        ]
        weaknesses = [
            f"{name} (mean {value:.2f})"
            for name, value in sorted(criterion_means.items())
            if value <= WEAKNESS_THRESHOLD
        ]

        scorecards.append(
            ModelScorecard(
                provider_name=provider_name,
                mean_weighted=sum(totals) / len(totals) if totals else 0.0,
                pass_rate=len(passes) / len(provider_scores) if provider_scores else 0.0,
                tasks_scored=len(provider_scores),
                criterion_means=criterion_means,
                strengths=strengths,
                weaknesses=weaknesses,
                skipped_criteria=sorted(skipped),
            )
        )

    per_task: dict[str, dict[str, float]] = {}
    for score in scores:
        per_task.setdefault(score.task_id, {})[score.provider_name] = score.weighted_total

    best_by_task: dict[str, str] = {}
    for task_id, values in per_task.items():
        if not values:
            continue
        best_by_task[task_id] = max(values.items(), key=lambda item: (item[1], item[0]))[0]

    return EvaluationReport(
        scorecards=scorecards,
        per_task=per_task,
        best_by_task=best_by_task,
        task_count=len(per_task),
    )


def to_markdown(report: EvaluationReport) -> str:
    """Render the report as markdown."""
    lines: list[str] = []
    lines.append("# LLM Rubric Evaluation Report")
    lines.append("")
    lines.append(f"Tasks evaluated: {report.task_count}")
    lines.append(f"Models compared: {', '.join(report.provider_names)}")
    lines.append("")

    lines.append("## Model scorecard")
    lines.append("")
    lines.append("| Model | Mean weighted score | Pass rate | Tasks scored |")
    lines.append("| --- | --- | --- | --- |")
    for scorecard in report.scorecards:
        lines.append(
            f"| {scorecard.provider_name} | {scorecard.mean_weighted:.3f} | "
            f"{scorecard.pass_rate * 100:.1f} percent | {scorecard.tasks_scored} |"
        )
    lines.append("")

    lines.append("## Head to head by task")
    lines.append("")
    header = "| Task | " + " | ".join(report.provider_names) + " | Best |"
    separator = "| --- |" + " --- |" * (len(report.provider_names) + 1)
    lines.append(header)
    lines.append(separator)
    for task_id in sorted(report.per_task):
        values = report.per_task[task_id]
        cells = " | ".join(
            f"{values.get(name, float('nan')):.3f}" if name in values else "n/a"
            for name in report.provider_names
        )
        lines.append(f"| {task_id} | {cells} | {report.best_by_task.get(task_id, 'n/a')} |")
    lines.append("")

    lines.append("## Strengths and weaknesses")
    lines.append("")
    for scorecard in report.scorecards:
        lines.append(f"### {scorecard.provider_name}")
        lines.append("")
        lines.append(f"Mean weighted score: {scorecard.mean_weighted:.3f}")
        lines.append("")
        if scorecard.strengths:
            lines.append("Strengths, criteria at or above "
                         f"{STRENGTH_THRESHOLD:.2f}:")
            for item in scorecard.strengths:
                lines.append(f"- {item}")
        else:
            lines.append(
                f"Strengths: no criterion reached the {STRENGTH_THRESHOLD:.2f} threshold."
            )
        lines.append("")
        if scorecard.weaknesses:
            lines.append(
                f"Weaknesses, criteria at or below {WEAKNESS_THRESHOLD:.2f}:"
            )
            for item in scorecard.weaknesses:
                lines.append(f"- {item}")
        else:
            lines.append(
                f"Weaknesses: no criterion fell to the {WEAKNESS_THRESHOLD:.2f} threshold."
            )
        lines.append("")
        if scorecard.skipped_criteria:
            lines.append(
                "Criteria skipped because no matching grader ran: "
                + ", ".join(scorecard.skipped_criteria)
            )
            lines.append("")

    lines.append("## Criterion detail")
    lines.append("")
    criteria = sorted(
        {name for scorecard in report.scorecards for name in scorecard.criterion_means}
    )
    lines.append("| Model | " + " | ".join(criteria) + " |")
    lines.append("| --- |" + " --- |" * len(criteria))
    for scorecard in report.scorecards:
        cells = " | ".join(
            f"{scorecard.criterion_means[name]:.3f}" if name in scorecard.criterion_means else "n/a"
            for name in criteria
        )
        lines.append(f"| {scorecard.provider_name} | {cells} |")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def to_csv_rows(scores) -> list[dict]:
    """Flatten rubric scores into one row per model, task and criterion."""
    rows: list[dict] = []
    for score in scores:
        for item in score.criterion_scores:
            rows.append(
                {
                    "provider_name": score.provider_name,
                    "task_id": score.task_id,
                    "rubric_id": score.rubric_id,
                    "criterion": item.criterion_name,
                    "criterion_weight": item.weight,
                    "applicable": item.applicable,
                    "score": item.score,
                    "level": item.level,
                    "weighted_total": score.weighted_total,
                    "passed": score.passed,
                }
            )
    return rows
