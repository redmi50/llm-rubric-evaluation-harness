"""Pytest configuration.

Placing this file at the repository root puts the repository root on ``sys.path``
so that ``rubric_harness`` imports without an install step, and it gives the tests
a single shared set of tasks and rubrics loaded from disk.
"""

from __future__ import annotations

import pytest

from rubric_harness.rubric import load_rubrics
from rubric_harness.tasks import load_tasks


@pytest.fixture(scope="session")
def tasks():
    """The task bank, loaded once per session."""
    return load_tasks()


@pytest.fixture(scope="session")
def rubrics():
    """The rubric library, loaded once per session."""
    return load_rubrics()


@pytest.fixture(scope="session")
def tasks_by_id(tasks):
    """The task bank keyed by task id."""
    return {task.task_id: task for task in tasks}
