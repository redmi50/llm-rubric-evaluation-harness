"""Deterministic offline provider.

This provider exists so the harness can be demonstrated, tested and reviewed with
no API key and no network. It answers from a reference table keyed on the prompt,
then perturbs the reference in a way that matches its quality profile, which
simulates a model that is variously careful, sloppy or confidently wrong.

Every profile is a pure function of the prompt and the reference table. There is
no clock, no random source, and no hidden state, so two runs produce identical
text and identical scores.
"""

from __future__ import annotations

import hashlib
import re

PROFILES = (
    "exact",
    "off_by_small",
    "wrong_method",
    "verbose_correct",
    "partially_correct",
)

OFF_BY_SMALL_FACTOR = 1.02
WRONG_METHOD_FACTOR = 2.5

_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_CURRENCY = re.compile(r"[$£€]")

_VERBOSE_PREFIX = (
    "That is a good question and it is worth working through carefully. "
    "Let me lay out the reasoning step by step before giving the answer. "
)
_VERBOSE_SUFFIX = (
    " I hope that breakdown is useful. Please let me know if you would like the "
    "same working shown with different assumptions."
)
_WRONG_TEXT = (
    "The figures here look broadly consistent with expectations, so I would not "
    "read too much into the movement. No further investigation looks necessary at "
    "this stage."
)


def _fingerprint(text: str) -> int:
    """A stable integer fingerprint of the prompt, used to pick a behaviour."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _first_sentence(text: str) -> str:
    """Everything up to and including the first sentence end."""
    parts = [part for part in re.split(r"(?<=[.!?])\s+", str(text).strip()) if part]
    return parts[0] if parts else str(text).strip()


def _shift_number(text: str, factor: float) -> str:
    """Multiply the first number in the text by a factor, in place."""
    match = _NUMBER.search(text)
    if not match:
        return text
    original = match.group(0)
    try:
        value = float(original.replace(",", ""))
    except ValueError:
        return text
    shifted = value * factor
    if abs(shifted) >= 100:
        replacement = f"{shifted:,.2f}"
    else:
        replacement = f"{shifted:.2f}"
    return text[: match.start()] + replacement + text[match.end() :]


def _drop_last_sentence(text: str) -> str:
    """Remove the final sentence, simulating an incomplete answer."""
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]
    if len(sentences) < 2:
        return text
    return " ".join(sentences[:-1])


class DeterministicMockProvider:
    """An offline provider with a fixed, documented quality profile."""

    def __init__(
        self,
        profile: str = "exact",
        answers: dict[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile {profile!r}. Expected one of {PROFILES}.")
        self.profile = profile
        self.answers = dict(answers or {})
        self._name = name or f"mock-{profile}"
        self._calls = 0

    @property
    def name(self) -> str:
        """Short identifier used in reports."""
        return self._name

    @property
    def calls(self) -> int:
        """How many times this instance has been asked for a completion."""
        return self._calls

    def complete(self, prompt: str) -> str:
        """Return a deterministic response derived from the prompt."""
        self._calls += 1
        reference = self.answers.get(prompt)
        if reference is None:
            return (
                "I do not have a prepared answer for this prompt. "
                "This mock provider only answers prompts present in its reference table."
            )

        if self.profile == "exact":
            return reference
        if self.profile == "verbose_correct":
            return f"{_VERBOSE_PREFIX}{reference}{_VERBOSE_SUFFIX}"
        if self.profile == "off_by_small":
            return self._off_by_small(reference)
        if self.profile == "wrong_method":
            return self._wrong(reference)
        return self._partially_correct(prompt, reference)

    def _off_by_small(self, reference: str) -> str:
        """A response that is nearly right, wrong in a minor way.

        A reference that is nothing but a figure is the answer, so it is shifted
        by a small factor. A reference written as prose is an explanation, where
        being slightly wrong means dropping the closing caveat rather than
        mangling an incidental number, so the body of the explanation survives.
        """
        if _NUMBER.fullmatch(reference.strip()):
            return _shift_number(reference, OFF_BY_SMALL_FACTOR)
        return _drop_last_sentence(reference)

    def _wrong(self, reference: str) -> str:
        """A confident answer built on the wrong method."""
        if _NUMBER.fullmatch(reference.strip()):
            return (
                "Working this through, the figure is "
                + _shift_number(reference, WRONG_METHOD_FACTOR)
                + "."
            )
        return _WRONG_TEXT

    def _partially_correct(self, prompt: str, reference: str) -> str:
        """Correct on roughly half of the prompts, wrong on the rest."""
        if _fingerprint(prompt) % 2 == 0:
            return reference
        if _NUMBER.fullmatch(reference.strip()):
            return _shift_number(reference, WRONG_METHOD_FACTOR)
        return _first_sentence(reference)


_REFERENCE_BLOCK = re.compile(
    r"Reference answer:\s*(.*?)\s*Response under review:", re.DOTALL
)
_RESPONSE_BLOCK = re.compile(
    r"Response under review:\s*(.*?)\s*Reply on two lines", re.DOTALL
)
_CRITERION_BLOCK = re.compile(
    r"Criterion:\s*(.*?)\s*What this criterion measures:", re.DOTALL
)
_DESCRIPTION_BLOCK = re.compile(
    r"What this criterion measures:\s*(.*?)\s*Level anchors:", re.DOTALL
)

# Criteria whose quality is a function of the extracted number rather than of the
# wording, so a response can be numerically wrong while still being well written.
_NUMERIC_CRITERIA = frozenset({"correctness", "answer_accuracy"})
# Criteria that measure the writing rather than the answer, where padding is the
# failure mode and a small numerical error is not.
_PRESENTATION_CRITERIA = frozenset({"clarity", "concision"})

# The band of response to reference length ratios that scores full marks, and the
# rate at which the score decays above it. A response at half the reference length
# scores 0.5 on proportion, and one at three times scores 0.3.
MIN_LENGTH_RATIO = 0.5
MAX_LENGTH_RATIO = 1.5
LENGTH_DECAY = 3.0

# Signal vocabulary per criterion family. The terms are the words a correct
# answer actually uses for that dimension, so they are read out of the reference
# answer rather than invented by the judge. The first family whose marker appears
# in the criterion name wins, which is why the more specific families come first.
_CUE_GROUPS: tuple[tuple[tuple[str, ...], frozenset[str]], ...] = (
    (
        ("assumption",),
        frozenset(
            {
                "adequate", "assumption", "assumptions", "balanced", "confound",
                "confounded", "confounder", "differs", "distribution", "expected",
                "independent", "independence", "mix", "normal", "power", "pre",
                "registered", "size", "unbalanced", "violated", "violation",
            }
        ),
    ),
    (
        ("transparency",),
        frozenset(
            {
                "defined", "definition", "exclusion", "exclusions", "metric",
                "period", "primary", "registered", "report", "reported", "sample",
                "source", "stated", "threshold", "window",
            }
        ),
    ),
    (
        ("interpretation",),
        frozenset(
            {
                "association", "causal", "causation", "conclusion", "decision",
                "effect", "estimate", "evidence", "inflated", "invalid", "matter",
                "practice", "real", "reproduce", "supported", "valid",
            }
        ),
    ),
    (
        ("reasoning",),
        frozenset(
            {
                "arises", "because", "consequence", "due", "hence", "means",
                "since", "therefore",
            }
        ),
    ),
    (
        ("decision",),
        frozenset(
            {
                "action", "confirm", "next", "owner", "recommend", "recommended",
                "restore", "review", "routing", "should",
            }
        ),
    ),
    (
        ("structure",),
        frozenset(
            {
                "cause", "concentrated", "driver", "effect", "fell", "increase",
                "led", "rose", "summary",
            }
        ),
    ),
    (
        ("method",),
        frozenset(
            {
                "adjust", "adjustment", "aggregate", "average", "base", "baseline",
                "bonferroni", "comparison", "comparisons", "corrected",
                "correction", "denominator", "mean", "model", "multiplicity",
                "opening", "pooled", "rate", "rebuild", "remove", "split",
                "standardised", "stratified", "test", "tests", "volume", "weighted",
            }
        ),
    ),
)


def _cue_terms(criterion: str) -> frozenset[str] | None:
    """The signal vocabulary for a criterion, or None if it has no family."""
    for markers, cues in _CUE_GROUPS:
        if any(marker in criterion for marker in markers):
            return cues
    return None


class DeterministicMockJudge:
    """An offline judge that emits a rubric level for a response.

    This is a stand-in for a model judge, not a model. It exists so the harness
    can be demonstrated, tested and reviewed with no API key and no network, and
    its scores are evidence about the mock profiles rather than about any real
    system.

    It reads the criterion, the reference answer and the response out of the
    judge prompt and scores one concrete signal, chosen by the criterion:

    * a criterion named ``correctness`` or ``answer_accuracy`` is scored on how
      close the response's number is to the reference's, with the gap expressed
      as a fraction of the reference so a rate, a count and a currency amount are
      treated alike;
    * a criterion named ``clarity`` or ``concision`` is scored on whether the
      response is proportionate to the reference, blended with the reference
      signal, so padding is penalised and a bare figure is not rewarded;
    * every other criterion is scored on the reference signal: how much of the
      reference's vocabulary the response retains, blended with how much of that
      criterion's signal vocabulary it keeps.

    The signal vocabulary is read out of the reference answer, so a response that
    is the reference scores full marks on every signal by construction, and the
    anchors in the rubric cannot distort the score of a correct answer. The
    result is raised to ``strictness``, which is how a stricter judge penalises
    partial answers more sharply and how two judges are made to disagree in a way
    worth measuring.

    There is no clock and no random source, so the same prompt always yields the
    same level.
    """

    def __init__(self, strictness: float = 1.0, name: str = "mock-judge") -> None:
        if strictness < 1.0:
            raise ValueError("strictness must be at least 1.0")
        self.strictness = strictness
        self._name = name
        self._calls = 0

    @property
    def name(self) -> str:
        """Short identifier used in reports."""
        return self._name

    @property
    def calls(self) -> int:
        """How many times this instance has been asked for a completion."""
        return self._calls

    def _blocks(self, prompt: str) -> tuple[str, str, str, str]:
        """Pull the criterion, description, reference and response out of a prompt."""
        criterion = _CRITERION_BLOCK.search(prompt)
        description = _DESCRIPTION_BLOCK.search(prompt)
        reference = _REFERENCE_BLOCK.search(prompt)
        response = _RESPONSE_BLOCK.search(prompt)
        return (
            criterion.group(1).strip().lower() if criterion else "",
            description.group(1).strip().lower() if description else "",
            reference.group(1) if reference else "",
            response.group(1) if response else "",
        )

    def reference_coverage(self, prompt: str) -> float:
        """Fraction of the reference's vocabulary that the response retains.

        A reference that tokenises to nothing, which is what a short figure such
        as ``3.5`` does, has no vocabulary to measure, so the comparison falls
        back to the words themselves: the response scores 1.0 when its text is the
        reference text and 0.0 otherwise.
        """
        _, _, reference, response = self._blocks(prompt)
        reference_tokens = set(_tokens(reference))
        if not reference_tokens:
            return 1.0 if response.split() == reference.split() else 0.0
        response_tokens = set(_tokens(response))
        return len(reference_tokens & response_tokens) / len(reference_tokens)

    def signal_coverage(self, prompt: str) -> float:
        """Fraction of the criterion's signal vocabulary that the response keeps.

        The checklist is the criterion's signal vocabulary intersected with the
        words the reference answer actually uses. Intersecting with the reference
        is what keeps a correct answer from being penalised for vocabulary it was
        never expected to use: a word that the reference itself does not contain
        is not demanded of the response. When the intersection is empty, which
        happens when the reference is a bare figure, the reference coverage is
        returned instead so the criterion still has a signal.
        """
        criterion, _, reference, response = self._blocks(prompt)
        cues = _cue_terms(criterion)
        if cues is None:
            return self.reference_coverage(prompt)

        relevant = cues & set(_tokens(reference))
        if not relevant:
            return self.reference_coverage(prompt)

        response_tokens = set(_tokens(response))
        return len(relevant & response_tokens) / len(relevant)

    def numeric_closeness(self, prompt: str) -> float:
        """A 0 to 1 score for how close the response's number is to the reference's.

        The distance is expressed as a fraction of the reference value, so it does
        not depend on whether the answer is a rate, a count or a currency amount.
        An exact match scores 1.0. Being ten percent out scores 0.5. Being more
        than twenty percent out scores 0.0.
        """
        _, _, reference, response = self._blocks(prompt)
        reference_value = _first_number(reference)
        response_value = _first_number(response)
        if reference_value is None:
            return self.reference_coverage(prompt)
        if response_value is None:
            return 0.0
        scale = max(abs(reference_value), 1e-9)
        relative_error = abs(response_value - reference_value) / scale
        return max(0.0, 1.0 - relative_error / 0.2)

    def length_appropriateness(self, prompt: str) -> float:
        """A 0 to 1 score for whether the response is proportionate to the reference.

        A response between four tenths and 1.25 times the reference length scores
        full marks. A longer one is scored down for padding, and a much shorter
        one for being a bare figure, so both the verbose profile and an answer
        stripped to a number lose points here.
        """
        _, _, reference, response = self._blocks(prompt)
        reference_words = len(str(reference).split())
        response_words = len(str(response).split())
        if response_words == 0 or reference_words == 0:
            return 0.0
        ratio = response_words / reference_words
        if MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
            return 1.0
        if ratio < MIN_LENGTH_RATIO:
            return ratio / MIN_LENGTH_RATIO
        return max(0.0, 1.0 - (ratio - MAX_LENGTH_RATIO) / LENGTH_DECAY)

    def reference_signal(self, prompt: str) -> tuple[float, str]:
        """How closely the response reproduces the reference, and why.

        The two terms are vocabulary overlap with the reference and coverage of
        the criterion's signal vocabulary. Blending them means a response that
        reproduces the reference scores full marks on every criterion by
        construction, which is the property a stand-in for a real judge must have
        if its scores are to be reported as evidence about the response rather
        than about the rubric's wording.
        """
        coverage = self.reference_coverage(prompt)
        signal = self.signal_coverage(prompt)
        blended = 0.5 * coverage + 0.5 * signal
        return blended, (
            f"reference coverage {coverage:.2f} blended with criterion signal "
            f"coverage {signal:.2f} gives {blended:.2f}"
        )

    def presentation_score(self, prompt: str) -> tuple[float, str]:
        """A 0 to 1 score for whether a response is readable and proportionate.

        Proportion is weighted by whether the response reproduces the reference at
        all, so a padded answer and a bare figure both lose points.
        """
        length = self.length_appropriateness(prompt)
        coverage = self.reference_coverage(prompt)
        raw = length * coverage
        return raw, (
            f"length appropriateness {length:.2f} weighted by reference coverage "
            f"{coverage:.2f} gives {raw:.2f}"
        )

    def criterion_score(self, prompt: str) -> tuple[float, str]:
        """Score the response on the criterion named in the prompt."""
        criterion, description, _, _ = self._blocks(prompt)
        haystack = f"{criterion} {description}"

        if any(name in haystack for name in _NUMERIC_CRITERIA):
            raw = self.numeric_closeness(prompt)
            return raw, f"numeric closeness {raw:.2f}"
        if any(name in haystack for name in _PRESENTATION_CRITERIA):
            raw, reason = self.presentation_score(prompt)
        else:
            raw, reason = self.reference_signal(prompt)

        adjusted = raw**self.strictness
        return adjusted, (
            f"{reason} adjusted by strictness {self.strictness:.2f} gives {adjusted:.2f}"
        )

    def complete(self, prompt: str) -> str:
        """Return a LEVEL and REASON pair for the criterion in the prompt."""
        self._calls += 1
        score, reason = self.criterion_score(prompt)
        level = min(max(int(round(1 + min(max(score, 0.0), 1.0) * 4)), 1), 5)
        return f"LEVEL: {level}\nREASON: {reason}."


def _first_number(text: str) -> float | None:
    """The first number in the text, with separators and currency removed."""
    if not text:
        return None
    cleaned = _CURRENCY.sub("", str(text))
    match = _NUMBER.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _tokens(text: str) -> list[str]:
    """Simple alphanumeric token list used by the mock judge."""
    return [token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if len(token) > 1]
