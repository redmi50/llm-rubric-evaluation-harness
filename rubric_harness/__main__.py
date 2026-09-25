"""Command line interface.

    python -m rubric_harness list-tasks
    python -m rubric_harness list-rubrics
    python -m rubric_harness run --out reports
    python -m rubric_harness report --run reports/run.json
    python -m rubric_harness agreement
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .agreement import agreement_matrix, overall_agreement, percent_agreement
from .graders import KeywordGrader, LLMJudgeGrader, NumericGrader
from .providers import DeterministicMockJudge, get_provider
from .report import build_report, to_csv_rows, to_markdown
from .rubric import load_rubrics
from .runner import ResponseRecord, run_matrix
from .scoring import evaluate_run
from .tasks import load_tasks

DEFAULT_PROVIDERS = "mock-exact,mock-off_by_small,mock-partially_correct,mock-wrong_method"
DEFAULT_OUT = "reports"
DEFAULT_CACHE = ".cache/responses"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="rubric_harness",
        description="Evaluate model responses against weighted rubrics.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-tasks", help="Print every task in the bank.")
    subparsers.add_parser("list-rubrics", help="Print every rubric and its criteria.")

    run_parser = subparsers.add_parser("run", help="Run the suite and grade it.")
    run_parser.add_argument(
        "--providers",
        default=DEFAULT_PROVIDERS,
        help="Comma separated provider names.",
    )
    run_parser.add_argument("--out", default=DEFAULT_OUT, help="Output directory.")
    run_parser.add_argument(
        "--cache-dir", default=DEFAULT_CACHE, help="Directory for the response cache."
    )
    run_parser.add_argument(
        "--no-cache", action="store_true", help="Bypass the response cache."
    )
    run_parser.add_argument(
        "--judge",
        action="store_true",
        help="Also grade the holistic criterion with a judge.",
    )
    run_parser.add_argument(
        "--judge-strictness",
        type=float,
        default=1.0,
        help="Strictness of the offline judge when --judge is used.",
    )
    run_parser.add_argument(
        "--judge-provider",
        default=None,
        help="Use a real provider as the judge, for example openai or anthropic.",
    )

    report_parser = subparsers.add_parser("report", help="Re-render a saved run.")
    report_parser.add_argument("--run", required=True, help="Path to run.json.")
    report_parser.add_argument("--out", default=None, help="Where to write the markdown.")

    agreement_parser = subparsers.add_parser(
        "agreement", help="Measure agreement between two judges for calibration."
    )
    agreement_parser.add_argument("--lenient-strictness", type=float, default=1.0)
    agreement_parser.add_argument("--strict-strictness", type=float, default=2.0)

    return parser


def cmd_list_tasks() -> int:
    """Print the task bank."""
    tasks = load_tasks()
    print(f"{len(tasks)} tasks\n")
    for task in tasks:
        print(f"{task.task_id}  [{task.answer_type}]  rubric={task.rubric_id}")
        print(f"  domain: {task.domain}")
        print(f"  prompt: {task.prompt}")
        print(f"  reference: {task.expected_reference}")
        print(f"  key facts: {len(task.key_facts)}")
        print()
    return 0


def cmd_list_rubrics() -> int:
    """Print the rubric library."""
    rubrics = load_rubrics()
    print(f"{len(rubrics)} rubrics\n")
    for rubric_id in sorted(rubrics):
        rubric = rubrics[rubric_id]
        print(f"{rubric.rubric_id}: {rubric.title}")
        print(f"  pass threshold: {rubric.pass_threshold}")
        for criterion in rubric.criteria:
            print(
                f"  - {criterion.name} (weight {criterion.weight}, "
                f"grader {criterion.grader})"
            )
        print()
    return 0


def _resolve_judge(args):
    if args.judge_provider:
        return get_provider(args.judge_provider)
    return DeterministicMockJudge(args.judge_strictness, name="offline-judge")


def cmd_run(args) -> int:
    """Run the suite, grade it and write the artefacts."""
    tasks = load_tasks()
    rubrics = load_rubrics()
    tasks_by_id = {task.task_id: task for task in tasks}

    provider_names = [name.strip() for name in args.providers.split(",") if name.strip()]
    if not provider_names:
        print("No providers requested.", file=sys.stderr)
        return 2

    providers = [get_provider(name, tasks) for name in provider_names]

    graders = [NumericGrader(), KeywordGrader()]
    if args.judge:
        graders.append(LLMJudgeGrader(_resolve_judge(args)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = run_matrix(
        tasks,
        providers,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    scores = evaluate_run(records, tasks_by_id, rubrics, graders)
    report = build_report(scores)

    markdown_path = out_dir / "model_scorecard.md"
    markdown_path.write_text(to_markdown(report), encoding="utf-8")

    csv_path = out_dir / "results.csv"
    rows = to_csv_rows(scores)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    run_path = out_dir / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "tasks": [task.task_id for task in tasks],
                "records": [record.as_dict() for record in records],
                "scores": [score.as_dict() for score in scores],
                "report": report.as_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cached = sum(1 for record in records if record.cached)
    print(f"Tasks: {len(tasks)}")
    print(f"Providers: {', '.join(provider_names)}")
    print(f"Responses: {len(records)}, served from cache: {cached}")
    print(f"Criteria skipped for lack of a grader: "
          f"{', '.join(sorted({name for scorecard in report.scorecards for name in scorecard.skipped_criteria})) or 'none'}")
    print()
    for scorecard in report.scorecards:
        print(
            f"{scorecard.provider_name:28s} mean {scorecard.mean_weighted:.3f}  "
            f"pass rate {scorecard.pass_rate * 100:.1f} percent"
        )
    print()
    print(f"Wrote {markdown_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {run_path}")
    return 0


def cmd_report(args) -> int:
    """Re-render a saved run from its JSON."""
    run_path = Path(args.run)
    if not run_path.is_file():
        print(f"Run file not found: {run_path}", file=sys.stderr)
        return 2

    payload = json.loads(run_path.read_text(encoding="utf-8"))

    from .scoring import CriterionScore, RubricScore

    scores: list[RubricScore] = []
    for item in payload.get("scores", []):
        score = RubricScore(
            task_id=item["task_id"],
            provider_name=item["provider_name"],
            rubric_id=item["rubric_id"],
            weighted_total=item["weighted_total"],
            passed=item["passed"],
            skipped_criteria=list(item.get("skipped_criteria", [])),
            notes=item.get("notes", ""),
        )
        score.criterion_scores = [
            CriterionScore(
                criterion_name=entry["criterion"],
                weight=entry["weight"],
                score=entry["score"],
                level=entry["level"],
                explanation=entry.get("explanation", ""),
                applicable=entry.get("applicable", True),
            )
            for entry in item.get("criteria", [])
        ]
        scores.append(score)

    report = build_report(scores)
    rendered = to_markdown(report)
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(rendered)
    return 0


def cmd_agreement(args) -> int:
    """Measure two offline judges against each other on the same responses.

    The judges see identical responses and are asked about the same criteria, so
    the only thing that differs between them is their strictness. Any disagreement
    in the matrix is therefore attributable to the rubric anchors being read
    differently rather than to the responses, which is exactly what a calibration
    session is trying to measure.
    """
    tasks = load_tasks()
    tasks_by_id = {task.task_id: task for task in tasks}
    rubrics = load_rubrics()
    providers = [
        get_provider("mock-exact", tasks),
        get_provider("mock-off_by_small", tasks),
        get_provider("mock-partially_correct", tasks),
        get_provider("mock-wrong_method", tasks),
    ]
    records = run_matrix(tasks, providers, use_cache=False)

    judges = {
        f"lenient-{args.lenient_strictness:.1f}": DeterministicMockJudge(
            args.lenient_strictness, name="lenient"
        ),
        f"strict-{args.strict_strictness:.1f}": DeterministicMockJudge(
            args.strict_strictness, name="strict"
        ),
    }

    label_sets: dict[str, list[int]] = {}
    for name, judge in judges.items():
        grader = LLMJudgeGrader(judge, name="holistic")
        levels: list[int] = []
        for record in records:
            task = tasks_by_id[record.task_id]
            rubric = rubrics[task.rubric_id]
            result = grader.grade_criterion(task, record.response_text, rubric.criteria[0])
            levels.append(int(result.details["level"]))
        label_sets[name] = levels

    names = sorted(label_sets)
    observed = percent_agreement(label_sets[names[0]], label_sets[names[1]])
    matrix = agreement_matrix(label_sets)

    print(f"Items graded by both judges: {len(label_sets[names[0]])}")
    print(f"Percent agreement: {observed * 100:.1f} percent")
    print(f"Overall kappa: {overall_agreement(label_sets):.3f}")
    print()
    print("Pairwise kappa:")
    header = "  " + " ".join(f"{name:>16s}" for name in names)
    print(header)
    for left in names:
        cells = " ".join(f"{matrix[left][right]:16.3f}" for right in names)
        print(f"  {left:<14s} {cells}")
    print()
    print("Level distribution:")
    for name in names:
        counts: dict[int, int] = {}
        for level in label_sets[name]:
            counts[level] = counts.get(level, 0) + 1
        summary = ", ".join(f"level {level}: {counts[level]}" for level in sorted(counts))
        print(f"  {name}: {summary}")
    return 0


def main(argv=None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-tasks":
        return cmd_list_tasks()
    if args.command == "list-rubrics":
        return cmd_list_rubrics()
    if args.command == "run":
        return cmd_run(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "agreement":
        return cmd_agreement(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
