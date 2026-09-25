"""The three graders, tested against hand-checked inputs."""

from __future__ import annotations

import pytest

from rubric_harness.graders import (
    GraderResult,
    KeywordGrader,
    NumericGrader,
    default_graders,
    grader_names,
)
from rubric_harness.graders.keyword import fact_present, normalise, tokenize
from rubric_harness.graders.numeric import extract_number, parse_reference
from rubric_harness.tasks import Task


def numeric_task(reference: str, tolerance: float = 0.5) -> Task:
    return Task(
        task_id="numeric",
        domain="d",
        answer_type="numeric",
        prompt="p",
        expected_reference=reference,
        rubric_id="r",
        tolerance=tolerance,
    )


def text_task(reference: str, key_facts=()) -> Task:
    return Task(
        task_id="text",
        domain="d",
        answer_type="text",
        prompt="p",
        expected_reference=reference,
        rubric_id="r",
        key_facts=tuple(key_facts),
    )


class TestGraderResult:
    def test_a_score_above_one_is_rejected(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            GraderResult(grader_name="g", score=1.2, explanation="")

    def test_a_negative_score_is_rejected(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            GraderResult(grader_name="g", score=-0.1, explanation="")


class TestNumberExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("136.07", 136.07),
            ("The answer is 136.07.", 136.07),
            ("1,234.50", 1234.50),
            ("$1,234.50", 1234.50),
            ("3.5 percent", 3.5),
            ("(1,200)", -1200.0),
            ("-42", -42.0),
            ("+7.5", 7.5),
            ("- 42", -42.0),
        ],
    )
    def test_extraction_handles_documented_formats(self, text, expected):
        assert extract_number(text) == pytest.approx(expected)

    def test_text_without_a_number_returns_none(self):
        assert extract_number("no numbers here") is None

    def test_none_input_returns_none(self):
        assert extract_number(None) is None

    def test_a_parenthesised_number_is_negative_even_when_signed(self):
        assert extract_number("(1,200)") == -1200.0

    def test_parse_reference_reads_a_bare_figure(self):
        assert parse_reference("3.5") == pytest.approx(3.5)


class TestNumericGrader:
    def test_a_non_numeric_task_is_declined(self):
        assert NumericGrader().grade(text_task("ref"), "ref") is None

    def test_an_exact_answer_scores_one(self):
        result = NumericGrader().grade(numeric_task("136.07"), "The answer is 136.07.")
        assert result.score == 1.0

    def test_an_answer_within_tolerance_scores_one(self):
        result = NumericGrader().grade(numeric_task("136.07", 0.5), "136.20")
        assert result.score == 1.0

    def test_an_answer_within_ten_times_tolerance_scores_a_half(self):
        result = NumericGrader().grade(numeric_task("100.0", 0.5), "103.0")
        assert result.score == 0.5

    def test_an_answer_far_outside_tolerance_scores_zero(self):
        result = NumericGrader().grade(numeric_task("100.0", 0.5), "250.0")
        assert result.score == 0.0

    def test_an_unparsable_answer_scores_zero_and_says_so(self):
        result = NumericGrader().grade(numeric_task("100.0"), "I am not sure.")
        assert result.score == 0.0
        assert "No numeric value" in result.explanation
        assert result.details["observed"] is None

    def test_details_record_the_error_and_the_tolerance(self):
        result = NumericGrader().grade(numeric_task("100.0", 0.5), "100.2")
        assert result.details["absolute_error"] == pytest.approx(0.2)
        assert result.details["tolerance"] == 0.5

    def test_a_numeric_task_with_an_unparsable_reference_is_a_configuration_error(self):
        with pytest.raises(ValueError, match="contains no number"):
            NumericGrader().grade(numeric_task("not a number"), "5")


class TestNormalisation:
    @pytest.mark.parametrize(
        "left,right",
        [
            ("remove", "removed"),
            ("remove", "removing"),
            ("remove", "removes"),
            ("split", "splits"),
            ("rate", "rates"),
        ],
    )
    def test_inflected_forms_share_a_stem(self, left, right):
        assert normalise(left) == normalise(right)

    def test_a_double_s_is_not_stripped(self):
        assert normalise("class") == "class"

    def test_stopwords_are_dropped(self):
        assert "the" not in tokenize("the answer")

    def test_single_characters_are_dropped(self):
        assert tokenize("a b c") == []


class TestKeywordGrader:
    def test_a_task_without_key_facts_is_declined(self):
        assert KeywordGrader().grade(text_task("ref"), "ref") is None

    def test_full_coverage_scores_one(self):
        task = text_task("the field leaks the target", ["the field leaks the target"])
        assert KeywordGrader().grade(task, "the field leaks the target").score == 1.0

    def test_partial_coverage_scores_the_fraction(self):
        task = text_task("ref", ["alpha beta", "gamma delta"])
        result = KeywordGrader().grade(task, "alpha beta only")
        assert result.score == 0.5
        assert "gamma delta" in result.explanation

    def test_no_coverage_scores_zero(self):
        task = text_task("ref", ["alpha beta"])
        result = KeywordGrader().grade(task, "nothing relevant")
        assert result.score == 0.0
        assert "Covered none" in result.explanation

    def test_inflected_wording_still_matches(self):
        task = text_task("ref", ["the analyst removes the leaked feature"])
        assert KeywordGrader().grade(task, "the analyst removed the leaking feature").score == 1.0

    def test_details_list_what_was_covered_and_missed(self):
        task = text_task("ref", ["alpha beta", "gamma delta"])
        details = KeywordGrader().grade(task, "alpha beta only").details
        assert details["covered"] == ["alpha beta"]
        assert details["missing"] == ["gamma delta"]
        assert details["total"] == 2

    def test_fact_present_requires_every_token(self):
        assert not fact_present("alpha beta", {"alpha"})
        assert fact_present("alpha beta", {"alpha", "beta"})


class TestDefaultGraders:
    def test_the_offline_set_excludes_the_judge(self):
        names = [grader.name for grader in default_graders()]
        assert names == ["answer_accuracy", "key_facts"]

    def test_supplying_a_judge_provider_adds_the_holistic_grader(self):
        names = [grader.name for grader in default_graders(object())]
        assert "holistic" in names

    def test_grader_names_lists_the_three_keys(self):
        assert set(grader_names()) == {"answer_accuracy", "key_facts", "holistic"}
