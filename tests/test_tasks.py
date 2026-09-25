"""Task bank loading and validation."""

from __future__ import annotations

import pytest

from rubric_harness.tasks import Task, TaskValidationError, load_tasks


def write_tasks(tmp_path, body: str):
    directory = tmp_path / "tasks"
    directory.mkdir()
    (directory / "suite.yaml").write_text(body, encoding="utf-8")
    return tmp_path


class TestTask:
    def test_a_numeric_task_knows_it_is_numeric(self):
        task = Task(
            task_id="t",
            domain="d",
            answer_type="numeric",
            prompt="p",
            expected_reference="1.0",
            rubric_id="r",
            tolerance=0.1,
        )
        assert task.is_numeric

    def test_a_text_task_is_not_numeric(self):
        task = Task(
            task_id="t",
            domain="d",
            answer_type="text",
            prompt="p",
            expected_reference="ref",
            rubric_id="r",
        )
        assert not task.is_numeric

    def test_an_unknown_answer_type_is_rejected(self):
        with pytest.raises(TaskValidationError, match="answer_type"):
            Task(
                task_id="t",
                domain="d",
                answer_type="essay",
                prompt="p",
                expected_reference="ref",
                rubric_id="r",
            )

    def test_a_negative_tolerance_is_rejected(self):
        with pytest.raises(TaskValidationError, match="non-negative tolerance"):
            Task(
                task_id="t",
                domain="d",
                answer_type="text",
                prompt="p",
                expected_reference="ref",
                rubric_id="r",
                tolerance=-1.0,
            )

    def test_a_blank_prompt_is_rejected(self):
        with pytest.raises(TaskValidationError, match="non-empty prompt"):
            Task(
                task_id="t",
                domain="d",
                answer_type="text",
                prompt="  ",
                expected_reference="ref",
                rubric_id="r",
            )


class TestTaskLoading:
    def test_a_numeric_task_without_tolerance_is_rejected(self, tmp_path):
        root = write_tasks(
            tmp_path,
            "tasks:\n"
            "  - task_id: t\n"
            "    answer_type: numeric\n"
            "    prompt: p\n"
            "    expected_reference: '1.0'\n"
            "    rubric_id: r\n",
        )
        with pytest.raises(TaskValidationError, match="positive tolerance"):
            load_tasks(root)

    def test_missing_required_keys_are_reported(self, tmp_path):
        root = write_tasks(tmp_path, "tasks:\n  - task_id: t\n    prompt: p\n")
        with pytest.raises(TaskValidationError, match="missing required keys"):
            load_tasks(root)

    def test_an_empty_task_list_is_rejected(self, tmp_path):
        root = write_tasks(tmp_path, "tasks: []\n")
        with pytest.raises(TaskValidationError, match="non-empty 'tasks' list"):
            load_tasks(root)

    def test_key_facts_must_be_a_list(self, tmp_path):
        root = write_tasks(
            tmp_path,
            "tasks:\n"
            "  - task_id: t\n"
            "    answer_type: text\n"
            "    prompt: p\n"
            "    expected_reference: ref\n"
            "    rubric_id: r\n"
            "    key_facts: not-a-list\n",
        )
        with pytest.raises(TaskValidationError, match="key_facts must be a list"):
            load_tasks(root)

    def test_duplicate_ids_across_files_are_rejected(self, tmp_path):
        directory = tmp_path / "tasks"
        directory.mkdir()
        body = (
            "tasks:\n"
            "  - task_id: t\n"
            "    answer_type: text\n"
            "    prompt: p\n"
            "    expected_reference: ref\n"
            "    rubric_id: r\n"
        )
        (directory / "a.yaml").write_text(body, encoding="utf-8")
        (directory / "b.yaml").write_text(body, encoding="utf-8")
        with pytest.raises(TaskValidationError, match="Duplicate task ids"):
            load_tasks(tmp_path)

    def test_a_missing_directory_is_reported(self, tmp_path):
        with pytest.raises(TaskValidationError, match="Tasks directory not found"):
            load_tasks(tmp_path)


class TestShippedTasks:
    def test_the_bank_is_not_empty(self, tasks):
        assert len(tasks) >= 6

    def test_every_task_points_at_a_shipped_rubric(self, tasks, rubrics):
        for task in tasks:
            assert task.rubric_id in rubrics

    def test_task_ids_are_unique(self, tasks):
        identifiers = [task.task_id for task in tasks]
        assert len(identifiers) == len(set(identifiers))

    def test_numeric_tasks_declare_a_tolerance_and_a_parsable_reference(self, tasks):
        from rubric_harness.graders.numeric import parse_reference

        numeric = [task for task in tasks if task.is_numeric]
        assert numeric, "the bank is expected to contain at least one numeric task"
        for task in numeric:
            assert task.tolerance > 0
            assert parse_reference(task.expected_reference) is not None

    def test_the_bank_covers_both_answer_types(self, tasks):
        types = {task.answer_type for task in tasks}
        assert types == {"numeric", "text"}

    def test_key_facts_are_only_on_text_tasks(self, tasks):
        for task in tasks:
            if task.is_numeric:
                assert task.key_facts == ()

    def test_every_declared_key_fact_is_met_by_the_reference_answer(self, tasks):
        """The reference is the standard a response is measured against.

        A fact the reference itself does not satisfy would make full marks
        unreachable, so this is checked rather than assumed.
        """
        from rubric_harness.graders.keyword import fact_present, tokenize

        for task in tasks:
            if not task.key_facts:
                continue
            reference_tokens = set(tokenize(task.expected_reference))
            for fact in task.key_facts:
                assert fact_present(fact, reference_tokens), (
                    f"{task.task_id}: key fact not covered by its own reference: {fact!r}"
                )
