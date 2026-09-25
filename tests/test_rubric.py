"""Rubric definition, validation and loading."""

from __future__ import annotations

import pytest

from rubric_harness.rubric import (
    Criterion,
    Rubric,
    RubricValidationError,
    load_rubrics,
)

LEVELS = {level: f"Descriptor for level {level}." for level in (1, 2, 3, 4, 5)}


def make_criterion(**overrides) -> Criterion:
    payload = {"name": "method", "weight": 1.0, "description": "d", "levels": LEVELS}
    payload.update(overrides)
    return Criterion(**payload)


class TestCriterion:
    def test_grader_defaults_to_the_criterion_name(self):
        assert make_criterion(name="clarity").grader == "clarity"

    def test_an_explicit_grader_is_kept(self):
        assert make_criterion(grader="key_facts").grader == "key_facts"

    def test_a_blank_name_is_rejected(self):
        with pytest.raises(RubricValidationError, match="non-empty name"):
            make_criterion(name="   ")

    def test_a_non_positive_weight_is_rejected(self):
        with pytest.raises(RubricValidationError, match="positive weight"):
            make_criterion(weight=0.0)

    def test_a_missing_level_is_rejected(self):
        with pytest.raises(RubricValidationError, match="missing descriptors"):
            make_criterion(levels={1: "a", 2: "b"})

    def test_descriptor_returns_the_anchor_text(self):
        assert make_criterion().descriptor(3) == "Descriptor for level 3."

    def test_descriptor_rejects_an_unknown_level(self):
        with pytest.raises(RubricValidationError, match="no descriptor"):
            make_criterion().descriptor(6)

    def test_from_dict_coerces_string_level_keys(self):
        criterion = Criterion.from_dict(
            {
                "name": "clarity",
                "weight": "0.25",
                "description": "d",
                "levels": {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e"},
            }
        )
        assert criterion.weight == 0.25
        assert sorted(criterion.levels) == [1, 2, 3, 4, 5]

    def test_from_dict_rejects_a_non_numeric_level_key(self):
        with pytest.raises(RubricValidationError, match="non-numeric level key"):
            Criterion.from_dict(
                {
                    "name": "clarity",
                    "weight": 1.0,
                    "description": "d",
                    "levels": {"one": "a"},
                }
            )


class TestRubric:
    def make_rubric(self, weights=(0.6, 0.4)) -> Rubric:
        return Rubric(
            rubric_id="example",
            title="Example",
            criteria=tuple(
                make_criterion(name=f"c{index}", weight=weight)
                for index, weight in enumerate(weights)
            ),
        )

    def test_weights_must_sum_to_one(self):
        with pytest.raises(RubricValidationError, match="sum to"):
            self.make_rubric(weights=(0.6, 0.5))

    def test_a_rubric_needs_at_least_one_criterion(self):
        with pytest.raises(RubricValidationError, match="no criteria"):
            Rubric(rubric_id="example", title="Example", criteria=())

    def test_duplicate_criterion_names_are_rejected(self):
        with pytest.raises(RubricValidationError, match="duplicate criterion names"):
            Rubric(
                rubric_id="example",
                title="Example",
                criteria=(
                    make_criterion(name="same", weight=0.5),
                    make_criterion(name="same", weight=0.5),
                ),
            )

    def test_an_out_of_range_pass_threshold_is_rejected(self):
        with pytest.raises(RubricValidationError, match="pass_threshold"):
            Rubric(
                rubric_id="example",
                title="Example",
                criteria=(make_criterion(weight=1.0),),
                pass_threshold=1.5,
            )

    def test_get_fetches_a_criterion_by_name(self):
        assert self.make_rubric().get("c1").name == "c1"

    def test_get_rejects_an_unknown_name(self):
        with pytest.raises(KeyError, match="nope"):
            self.make_rubric().get("nope")

    def test_to_dict_round_trips_the_criterion_names(self):
        rubric = self.make_rubric()
        payload = rubric.to_dict()
        rebuilt = Rubric.from_dict(payload)
        assert rebuilt.criterion_names == rubric.criterion_names

    def test_from_yaml_rejects_a_mismatched_file_name(self, tmp_path):
        path = tmp_path / "wrong_name.yaml"
        path.write_text(
            "rubric_id: example\n"
            "title: Example\n"
            "criteria:\n"
            "  - name: only\n"
            "    weight: 1.0\n"
            "    description: d\n"
            "    levels: {1: a, 2: b, 3: c, 4: d, 5: e}\n",
            encoding="utf-8",
        )
        with pytest.raises(RubricValidationError, match="Expected the file to be named"):
            Rubric.from_yaml(path)

    def test_from_yaml_reports_invalid_yaml(self, tmp_path):
        path = tmp_path / "example.yaml"
        path.write_text("rubric_id: [unclosed\n", encoding="utf-8")
        with pytest.raises(RubricValidationError, match="not valid YAML"):
            Rubric.from_yaml(path)


class TestShippedRubrics:
    def test_the_library_loads(self, rubrics):
        assert len(rubrics) >= 4

    def test_every_shipped_rubric_is_valid(self, rubrics):
        for rubric in rubrics.values():
            rubric.validate()

    def test_every_rubric_file_is_named_after_its_id(self, rubrics):
        for rubric_id, rubric in rubrics.items():
            assert rubric.rubric_id == rubric_id

    def test_every_criterion_names_a_grader(self, rubrics):
        for rubric in rubrics.values():
            for criterion in rubric.criteria:
                assert criterion.grader

    def test_missing_directory_is_reported(self, tmp_path):
        with pytest.raises(RubricValidationError, match="Rubrics directory not found"):
            load_rubrics(tmp_path)

    def test_empty_directory_is_reported(self, tmp_path):
        (tmp_path / "rubrics").mkdir()
        with pytest.raises(RubricValidationError, match="No rubric files found"):
            load_rubrics(tmp_path)
