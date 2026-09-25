"""The offline providers and the offline judge.

The judge tests are the important ones. They pin the property that made the
first version of the judge wrong: a correct answer must not be scored down for
using vocabulary the rubric never asked for, so a response that reproduces the
reference has to score full marks on every criterion.
"""

from __future__ import annotations

import pytest

from rubric_harness.graders import LLMJudgeGrader
from rubric_harness.providers import (
    PROFILES,
    DeterministicMockJudge,
    DeterministicMockProvider,
    get_provider,
)
from rubric_harness.providers.base import ProviderConfigurationError
from rubric_harness.rubric import Criterion
from rubric_harness.tasks import Task

ANCHORS = {level: f"Anchor for level {level}." for level in (1, 2, 3, 4, 5)}


def make_task(reference: str, answer_type: str = "numeric") -> Task:
    return Task(
        task_id="t",
        domain="d",
        answer_type=answer_type,
        prompt="A question.",
        expected_reference=reference,
        rubric_id="r",
        tolerance=0.5,
    )


def make_criterion(name: str, description: str = "What it measures.") -> Criterion:
    return Criterion(name=name, weight=1.0, description=description, levels=ANCHORS)


def judge_prompt(reference: str, response: str, criterion: Criterion, task=None) -> str:
    task = task or make_task(reference)
    return LLMJudgeGrader(object()).build_prompt(task, response, criterion)


class TestMockProvider:
    def test_an_unknown_profile_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            DeterministicMockProvider("no-such-profile")

    def test_the_exact_profile_reproduces_the_reference(self):
        provider = DeterministicMockProvider("exact", {"q": "the answer"})
        assert provider.complete("q") == "the answer"

    def test_an_unprepared_prompt_says_so(self):
        provider = DeterministicMockProvider("exact", {})
        assert "no" in provider.complete("unknown").lower()

    def test_the_call_counter_tracks_completions(self):
        provider = DeterministicMockProvider("exact", {"q": "a"})
        provider.complete("q")
        provider.complete("q")
        assert provider.calls == 2

    def test_the_name_defaults_to_the_profile(self):
        assert DeterministicMockProvider("exact").name == "mock-exact"

    def test_an_explicit_name_is_kept(self):
        assert DeterministicMockProvider("exact", name="mine").name == "mine"

    def test_the_same_prompt_yields_the_same_text_every_time(self):
        answers = {"q": "136.07"}
        first = DeterministicMockProvider("partially_correct", answers).complete("q")
        second = DeterministicMockProvider("partially_correct", answers).complete("q")
        assert first == second

    def test_off_by_small_moves_a_bare_figure_but_keeps_it_parsable(self):
        provider = DeterministicMockProvider("off_by_small", {"q": "100.00"})
        assert provider.complete("q") == "102.00"

    def test_off_by_small_does_not_mangle_a_number_inside_prose(self):
        """A prose reference is an explanation, so the leading figure is intact."""
        reference = "Failures rose from 0.4 to 1.1 percent this week."
        provider = DeterministicMockProvider("off_by_small", {"q": reference})
        assert provider.complete("q").startswith("Failures rose from 0.4 to 1.1")

    def test_wrong_method_replaces_a_bare_figure_with_a_wrong_one(self):
        provider = DeterministicMockProvider("wrong_method", {"q": "100.00"})
        assert provider.complete("q") == "Working this through, the figure is 250.00."

    def test_wrong_method_replaces_prose_with_a_dismissal(self):
        provider = DeterministicMockProvider("wrong_method", {"q": "Some explanation."})
        assert "No further investigation" in provider.complete("q")

    def test_partially_correct_depends_on_the_prompt(self):
        provider = DeterministicMockProvider("partially_correct", {"q": "100.00"})
        response = provider.complete("q")
        assert response == "100.00" or response != "100.00"

    def test_every_profile_is_constructible(self):
        for profile in PROFILES:
            assert DeterministicMockProvider(profile, {}).profile == profile


class TestProviderRegistry:
    def test_mock_resolves_to_the_exact_profile(self):
        assert get_provider("mock").profile == "exact"

    def test_a_profiled_name_resolves_to_that_profile(self):
        assert get_provider("mock-off_by_small").profile == "off_by_small"

    def test_an_unknown_mock_profile_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown mock profile"):
            get_provider("mock-nonsense")

    def test_an_unknown_provider_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonsense")

    def test_a_real_provider_without_a_key_names_the_variable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderConfigurationError) as error:
            get_provider("openai")
        assert "OPENAI_API_KEY" in str(error.value)


class TestMockJudgeParsing:
    def test_a_strict_reply_is_parsed(self):
        level, reason = LLMJudgeGrader(object()).parse_level(
            "LEVEL: 4\nREASON: because it is clear."
        )
        assert level == 4
        assert reason == "because it is clear."

    def test_a_loose_reply_is_parsed(self):
        level, _ = LLMJudgeGrader(object()).parse_level("I would say 3 out of 5.")
        assert level == 3

    def test_an_unparsable_reply_yields_no_level(self):
        level, reason = LLMJudgeGrader(object()).parse_level("No idea.")
        assert level is None
        assert reason == "No idea."


class TestMockJudgeSignals:
    def test_a_reference_response_scores_full_marks_on_an_unknown_criterion(self):
        """The property the whole judge rests on: the reference must score 1.0."""
        reference = "Severity is a confounder because the mix differs by group."
        prompt = judge_prompt(reference, reference, make_criterion("reasoning"), make_task(reference, "text"))
        assert DeterministicMockJudge().criterion_score(prompt)[0] == pytest.approx(1.0)

    def test_a_reference_response_scores_full_marks_on_a_cue_criterion(self):
        reference = "The method must be corrected for multiplicity with a Bonferroni test."
        criterion = make_criterion("method_appropriateness")
        prompt = judge_prompt(reference, reference, criterion, make_task(reference, "text"))
        assert DeterministicMockJudge().criterion_score(prompt)[0] == pytest.approx(1.0)

    def test_a_reference_response_scores_full_marks_on_a_presentation_criterion(self):
        reference = "The answer is 136.07, computed as a volume weighted mean."
        criterion = make_criterion("clarity")
        prompt = judge_prompt(reference, reference, criterion, make_task(reference, "text"))
        assert DeterministicMockJudge().criterion_score(prompt)[0] == pytest.approx(1.0)

    def test_an_exact_bare_figure_scores_full_marks_on_the_numeric_criterion(self):
        criterion = make_criterion("correctness")
        prompt = judge_prompt("3.5", "3.5", criterion)
        assert DeterministicMockJudge().criterion_score(prompt)[0] == pytest.approx(1.0)

    def test_a_bare_figure_that_is_wrong_scores_zero_on_the_numeric_criterion(self):
        criterion = make_criterion("correctness")
        prompt = judge_prompt("3.5", "8.75", criterion)
        assert DeterministicMockJudge().criterion_score(prompt)[0] == 0.0

    def test_a_numeric_error_of_one_percent_still_scores_high(self):
        criterion = make_criterion("correctness")
        prompt = judge_prompt("100.00", "101.00", criterion)
        score = DeterministicMockJudge().criterion_score(prompt)[0]
        assert 0.9 < score < 1.0

    def test_a_numeric_error_of_ten_percent_scores_a_half(self):
        """The documented midpoint of the numeric scale."""
        criterion = make_criterion("correctness")
        prompt = judge_prompt("100.00", "110.00", criterion)
        assert DeterministicMockJudge().criterion_score(prompt)[0] == pytest.approx(0.5)

    def test_a_response_that_drops_the_substance_loses_the_signal(self):
        reference = "The feature leaks the target and must be removed before rebuilding."
        criterion = make_criterion("method")
        prompt = judge_prompt(reference, "", criterion, make_task(reference, "text"))
        assert DeterministicMockJudge().criterion_score(prompt)[0] == 0.0

    def test_padding_is_penalised_on_concision(self):
        reference = "The estimate should be reconciled to the ledger."
        padded = reference + " " + "This is extra commentary. " * 20
        criterion = make_criterion("concision")
        plain = DeterministicMockJudge().criterion_score(
            judge_prompt(reference, reference, criterion, make_task(reference, "text"))
        )[0]
        verbose = DeterministicMockJudge().criterion_score(
            judge_prompt(reference, padded, criterion, make_task(reference, "text"))
        )[0]
        assert verbose < plain

    def test_a_short_reference_is_not_penalised_for_being_short(self):
        """A one-word reference is complete, so length does not count against it."""
        criterion = make_criterion("clarity")
        prompt = judge_prompt("3.5", "3.5", criterion)
        assert DeterministicMockJudge().criterion_score(prompt)[0] == pytest.approx(1.0)

    def test_different_criteria_can_score_the_same_response_differently(self):
        reference = "The method must be corrected for multiplicity."
        response = "The method is fine as it stands."
        method = DeterministicMockJudge().criterion_score(
            judge_prompt(reference, response, make_criterion("method"), make_task(reference, "text"))
        )[0]
        concision = DeterministicMockJudge().criterion_score(
            judge_prompt(reference, response, make_criterion("concision"), make_task(reference, "text"))
        )[0]
        assert method != concision

    def test_the_same_prompt_always_yields_the_same_level(self):
        reference = "Alpha beta gamma."
        criterion = make_criterion("structure")
        prompt = judge_prompt(reference, reference, criterion, make_task(reference, "text"))
        judge = DeterministicMockJudge()
        assert judge.complete(prompt) == judge.complete(prompt)

    def test_reason_text_reports_the_signals_used(self):
        reference = "The method must be corrected for multiplicity."
        criterion = make_criterion("method")
        prompt = judge_prompt(reference, reference, criterion, make_task(reference, "text"))
        assert "reference coverage" in DeterministicMockJudge().criterion_score(prompt)[1]


class TestMockJudgeLevels:
    def test_reply_shape_is_level_then_reason(self):
        reference = "3.5"
        criterion = make_criterion("correctness")
        prompt = judge_prompt(reference, reference, criterion)
        reply = DeterministicMockJudge().complete(prompt)
        assert reply.startswith("LEVEL: 5")
        assert "\nREASON: " in reply

    def test_a_strictness_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1.0"):
            DeterministicMockJudge(0.5)

    def test_a_higher_strictness_never_increases_a_partial_score(self):
        reference = "The method must be corrected for multiplicity and checked for power."
        response = "The method must be corrected for multiplicity."
        criterion = make_criterion("method")
        task = make_task(reference, "text")
        lenient = LLMJudgeGrader(DeterministicMockJudge(1.0)).grade_criterion(
            task, response, criterion
        )
        strict = LLMJudgeGrader(DeterministicMockJudge(2.0)).grade_criterion(
            task, response, criterion
        )
        assert strict.details["level"] <= lenient.details["level"]

    def test_a_perfect_answer_survives_a_strict_judge(self):
        reference = "The method must be corrected for multiplicity."
        criterion = make_criterion("method")
        task = make_task(reference, "text")
        strict = LLMJudgeGrader(DeterministicMockJudge(3.0)).grade_criterion(
            task, reference, criterion
        )
        assert strict.details["level"] == 5

    def test_the_call_counter_tracks_completions(self):
        judge = DeterministicMockJudge()
        judge.complete(judge_prompt("3.5", "3.5", make_criterion("correctness")))
        assert judge.calls == 1


class TestLLMJudgeGrader:
    def test_the_prompt_carries_the_criterion_and_the_anchors(self):
        criterion = make_criterion("clarity", "Whether a reader can follow it.")
        prompt = LLMJudgeGrader(object()).build_prompt(make_task("3.5"), "3.5", criterion)
        assert "Criterion: clarity" in prompt
        assert "Whether a reader can follow it." in prompt
        assert "Anchor for level 5." in prompt
        assert "Reference answer: 3.5" in prompt
        assert "Response under review: 3.5" in prompt

    def test_without_a_criterion_a_default_stand_in_is_used(self):
        prompt = LLMJudgeGrader(object()).build_prompt(make_task("3.5"), "3.5")
        assert "overall quality" in prompt

    def test_a_missing_provider_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="requires a provider"):
            LLMJudgeGrader(None)

    def test_a_level_is_mapped_onto_the_zero_to_one_scale(self):
        class Fixed:
            name = "fixed"

            def complete(self, prompt):
                return "LEVEL: 5\nREASON: perfect."

        result = LLMJudgeGrader(Fixed()).grade_criterion(
            make_task("3.5"), "3.5", make_criterion("correctness")
        )
        assert result.score == 1.0
        assert result.details["level"] == 5
        assert result.details["parsed"] is True

    def test_the_result_is_keyed_on_the_grader_and_the_criterion(self):
        class Fixed:
            name = "fixed"

            def complete(self, prompt):
                return "LEVEL: 4\nREASON: good."

        result = LLMJudgeGrader(Fixed(), name="holistic").grade_criterion(
            make_task("3.5"), "3.5", make_criterion("clarity")
        )
        assert result.grader_name == "holistic:clarity"

    def test_an_unparsable_reply_is_recorded_not_guessed(self):
        class Broken:
            name = "broken"

            def complete(self, prompt):
                return "I decline to answer."

        grader = LLMJudgeGrader(Broken())
        result = grader.grade_criterion(make_task("3.5"), "3.5", make_criterion("clarity"))
        assert result.score == 0.5
        assert result.details["parsed"] is False
        assert "no parsable level" in result.explanation
        assert grader.unparsed == 1
        assert grader.parsed == 0

    def test_the_plain_grade_is_keyed_on_the_grader_name(self):
        class Fixed:
            name = "fixed"

            def complete(self, prompt):
                return "LEVEL: 3\nREASON: acceptable."

        assert LLMJudgeGrader(Fixed()).grade(make_task("3.5"), "3.5").grader_name == "holistic"
