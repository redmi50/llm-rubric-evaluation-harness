"""Rubric definition, validation and YAML loading.

A rubric is a set of weighted criteria. Every criterion defines a descriptor for
each level from 1 to 5, which is what makes the grading reproducible rather than
impressionistic: two graders applying the same rubric read the same anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
LEVELS = (1, 2, 3, 4, 5)
WEIGHT_TOLERANCE = 1e-6


class RubricValidationError(ValueError):
    """Raised when a rubric is internally inconsistent."""


@dataclass(frozen=True)
class Criterion:
    """One graded dimension of a rubric.

    The ``grader`` field names the grader that supplies the score, which is what
    lets a rubric drop a criterion cleanly when the matching grader is not part
    of the current run.
    """

    name: str
    weight: float
    description: str
    levels: dict[int, str] = field(default_factory=dict)
    grader: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise RubricValidationError("A criterion must have a non-empty name.")
        if self.weight <= 0:
            raise RubricValidationError(
                f"Criterion {self.name!r} must have a positive weight, got {self.weight}."
            )
        resolved_grader = self.grader or self.name
        object.__setattr__(self, "grader", resolved_grader)
        missing = [level for level in LEVELS if level not in self.levels]
        if missing:
            raise RubricValidationError(
                f"Criterion {self.name!r} is missing descriptors for levels "
                f"{missing}. Every level from 1 to 5 must be defined."
            )

    def descriptor(self, level: int) -> str:
        """Return the anchor text for a level."""
        if level not in self.levels:
            raise RubricValidationError(
                f"Criterion {self.name!r} has no descriptor for level {level}."
            )
        return self.levels[level]

    @classmethod
    def from_dict(cls, payload: dict) -> "Criterion":
        """Build a criterion from a decoded YAML mapping."""
        if not isinstance(payload, dict):
            raise RubricValidationError("A criterion must be a mapping.")
        raw_levels = payload.get("levels") or {}
        levels: dict[int, str] = {}
        for key, value in raw_levels.items():
            try:
                levels[int(key)] = str(value)
            except (TypeError, ValueError) as exc:
                raise RubricValidationError(
                    f"Criterion {payload.get('name')!r} has a non-numeric level key {key!r}."
                ) from exc
        return cls(
            name=str(payload.get("name", "")).strip(),
            weight=float(payload.get("weight", 0.0)),
            description=str(payload.get("description", "")).strip(),
            levels=levels,
            grader=str(payload.get("grader", "")).strip(),
        )


@dataclass(frozen=True)
class Rubric:
    """A weighted set of criteria plus a pass threshold."""

    rubric_id: str
    title: str
    criteria: tuple[Criterion, ...]
    pass_threshold: float = 0.6

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise RubricValidationError if the rubric is inconsistent."""
        if not self.rubric_id.strip():
            raise RubricValidationError("A rubric must have a non-empty rubric_id.")
        if not self.criteria:
            raise RubricValidationError(f"Rubric {self.rubric_id!r} has no criteria.")

        names = [criterion.name for criterion in self.criteria]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise RubricValidationError(
                f"Rubric {self.rubric_id!r} has duplicate criterion names: "
                f"{sorted(duplicates)}"
            )

        total = sum(criterion.weight for criterion in self.criteria)
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise RubricValidationError(
                f"Rubric {self.rubric_id!r} criterion weights sum to {total:.6f}, "
                "but they must sum to 1.0."
            )

        if not 0.0 <= self.pass_threshold <= 1.0:
            raise RubricValidationError(
                f"Rubric {self.rubric_id!r} pass_threshold must be between 0 and 1, "
                f"got {self.pass_threshold}."
            )

    @property
    def criterion_names(self) -> tuple[str, ...]:
        """Names of every criterion, in declaration order."""
        return tuple(criterion.name for criterion in self.criteria)

    def get(self, name: str) -> Criterion:
        """Fetch a criterion by name."""
        for criterion in self.criteria:
            if criterion.name == name:
                return criterion
        raise KeyError(f"No criterion named {name!r} in rubric {self.rubric_id!r}")

    @classmethod
    def from_dict(cls, payload: dict) -> "Rubric":
        """Build a rubric from a decoded YAML mapping."""
        if not isinstance(payload, dict):
            raise RubricValidationError("A rubric must be a mapping.")
        raw_criteria = payload.get("criteria") or []
        if not isinstance(raw_criteria, list):
            raise RubricValidationError("'criteria' must be a list.")
        return cls(
            rubric_id=str(payload.get("rubric_id", "")).strip(),
            title=str(payload.get("title", "")).strip(),
            criteria=tuple(Criterion.from_dict(item) for item in raw_criteria),
            pass_threshold=float(payload.get("pass_threshold", 0.6)),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Rubric":
        """Load a rubric from a YAML file."""
        path = Path(path)
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise RubricValidationError(f"{path.name} is not valid YAML: {exc}") from exc
        rubric = cls.from_dict(payload)
        if path.stem != rubric.rubric_id:
            raise RubricValidationError(
                f"{path.name} defines rubric_id {rubric.rubric_id!r}. "
                f"Expected the file to be named {rubric.rubric_id}.yaml."
            )
        return rubric

    def to_dict(self) -> dict:
        """Serializable representation."""
        return {
            "rubric_id": self.rubric_id,
            "title": self.title,
            "pass_threshold": self.pass_threshold,
            "criteria": [
                {
                    "name": criterion.name,
                    "weight": criterion.weight,
                    "grader": criterion.grader,
                    "description": criterion.description,
                    "levels": {str(k): v for k, v in sorted(criterion.levels.items())},
                }
                for criterion in self.criteria
            ],
        }


def load_rubrics(root: Path | str = DEFAULT_ROOT) -> dict[str, Rubric]:
    """Load every rubric YAML file under ``rubrics/``."""
    directory = Path(root) / "rubrics"
    if not directory.is_dir():
        raise RubricValidationError(f"Rubrics directory not found: {directory}")

    rubrics: dict[str, Rubric] = {}
    for path in sorted(directory.glob("*.yaml")):
        rubric = Rubric.from_yaml(path)
        rubrics[rubric.rubric_id] = rubric

    if not rubrics:
        raise RubricValidationError(f"No rubric files found in {directory}")
    return rubrics
