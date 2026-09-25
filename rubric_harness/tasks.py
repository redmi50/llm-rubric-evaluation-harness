"""Task bank loading.

A task pairs a prompt with the reference answer, the key facts a good answer must
contain, the rubric used to grade it, and the numeric tolerance for answers that
are expressed as a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
ANSWER_TYPES = ("numeric", "text")


class TaskValidationError(ValueError):
    """Raised when a task definition is malformed."""


@dataclass(frozen=True)
class Task:
    """One prompt in the evaluation suite."""

    task_id: str
    domain: str
    answer_type: str
    prompt: str
    expected_reference: str
    rubric_id: str
    key_facts: tuple[str, ...] = ()
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise TaskValidationError("A task must have a non-empty task_id.")
        if not self.prompt.strip():
            raise TaskValidationError(f"Task {self.task_id!r} must have a non-empty prompt.")
        if self.answer_type not in ANSWER_TYPES:
            raise TaskValidationError(
                f"Task {self.task_id!r} has answer_type {self.answer_type!r}. "
                f"Expected one of {ANSWER_TYPES}."
            )
        if self.tolerance < 0:
            raise TaskValidationError(
                f"Task {self.task_id!r} must have a non-negative tolerance."
            )

    @property
    def is_numeric(self) -> bool:
        """True when the reference answer is a number."""
        return self.answer_type == "numeric"

    @classmethod
    def from_dict(cls, payload: dict) -> "Task":
        """Build a task from a decoded YAML mapping."""
        if not isinstance(payload, dict):
            raise TaskValidationError("A task must be a mapping.")
        required = ("task_id", "prompt", "expected_reference", "rubric_id")
        missing = [key for key in required if not payload.get(key)]
        if missing:
            raise TaskValidationError(
                f"Task definition is missing required keys: {', '.join(missing)}"
            )
        answer_type = str(payload.get("answer_type", "text"))
        tolerance = float(payload.get("tolerance", 0.0))
        if answer_type == "numeric" and tolerance <= 0:
            raise TaskValidationError(
                f"Numeric task {payload['task_id']!r} must declare a positive tolerance."
            )
        key_facts = payload.get("key_facts") or []
        if not isinstance(key_facts, list):
            raise TaskValidationError(
                f"Task {payload['task_id']!r} key_facts must be a list."
            )
        return cls(
            task_id=str(payload["task_id"]),
            domain=str(payload.get("domain", "general")),
            answer_type=answer_type,
            prompt=str(payload["prompt"]).strip(),
            expected_reference=str(payload["expected_reference"]).strip(),
            rubric_id=str(payload["rubric_id"]),
            key_facts=tuple(str(fact) for fact in key_facts),
            tolerance=tolerance,
        )


def load_tasks(root: Path | str = DEFAULT_ROOT) -> tuple[Task, ...]:
    """Load every task from the YAML files under ``tasks/``."""
    directory = Path(root) / "tasks"
    if not directory.is_dir():
        raise TaskValidationError(f"Tasks directory not found: {directory}")

    tasks: list[Task] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise TaskValidationError(f"{path.name} is not valid YAML: {exc}") from exc
        entries = payload.get("tasks") if isinstance(payload, dict) else None
        if not isinstance(entries, list) or not entries:
            raise TaskValidationError(f"{path.name} must contain a non-empty 'tasks' list.")
        tasks.extend(Task.from_dict(entry) for entry in entries)

    if not tasks:
        raise TaskValidationError(f"No tasks found in {directory}")

    identifiers = [task.task_id for task in tasks]
    duplicates = {value for value in identifiers if identifiers.count(value) > 1}
    if duplicates:
        raise TaskValidationError(f"Duplicate task ids found: {sorted(duplicates)}")

    return tuple(sorted(tasks, key=lambda task: task.task_id))
