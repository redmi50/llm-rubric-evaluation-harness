"""Key fact coverage grading.

Scores the fraction of the task's key facts that appear in the response, using
normalised token matching so that punctuation, casing, simple plurals and the
-ing and -ed forms of a verb do not change the result. Without that
normalisation a grader measures exact word form rather than recall, and a
response that says "remove the field" would be marked as missing a fact written
as "the field must be removed".
"""

from __future__ import annotations

import re

from ..tasks import Task
from . import GRADER_KEYWORD, GraderResult

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
        "was", "were", "with",
    }
)


def normalise(token: str) -> str:
    """Reduce a word to a stem so inflected forms compare equal.

    The rules are deliberately few and stated in full: a trailing plural ``s``,
    a trailing ``ing``, a trailing ``ed`` and finally a trailing ``e`` are
    removed. The last rule is what makes the family agree, because ``removed``
    reduces to ``remov`` and ``remove`` has to reduce to the same token, not to
    ``remove``. They make ``remove``, ``removes``, ``removed`` and ``removing``
    one token. They are not a linguistic stemmer and do not map synonyms, so a
    fact written with different vocabulary from the reference is still a miss.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    if len(token) > 4 and token.endswith("ing"):
        token = token[:-3]
    if len(token) > 3 and token.endswith("ed"):
        token = token[:-2]
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords and normalise word forms."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    tokens: list[str] = []
    for raw in cleaned.split():
        if not raw or raw in _STOPWORDS or len(raw) < 2:
            continue
        tokens.append(normalise(raw))
    return tokens


def fact_present(fact: str, response_tokens: set[str]) -> bool:
    """True when every significant token of the fact appears in the response."""
    required = {token for token in tokenize(fact) if token not in _STOPWORDS}
    if not required:
        return False
    return required <= response_tokens


class KeywordGrader:
    """Grades how many of the task's key facts the response covers."""

    def __init__(self, name: str = GRADER_KEYWORD) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """The grader key used by rubrics."""
        return self._name

    def grade(self, task: Task, response_text: str) -> GraderResult | None:
        """Score key fact coverage, or return None when the task declares none."""
        if not task.key_facts:
            return None

        response_tokens = set(tokenize(response_text))
        covered = [fact for fact in task.key_facts if fact_present(fact, response_tokens)]
        missing = [fact for fact in task.key_facts if fact not in covered]

        score = len(covered) / len(task.key_facts)
        if covered:
            explanation = (
                f"Covered {len(covered)} of {len(task.key_facts)} key facts."
                + (f" Missing: {'; '.join(missing)}." if missing else "")
            )
        else:
            explanation = (
                f"Covered none of the {len(task.key_facts)} key facts. "
                f"Expected: {'; '.join(missing)}."
            )

        return GraderResult(
            grader_name=self.name,
            score=score,
            explanation=explanation,
            details={"covered": covered, "missing": missing, "total": len(task.key_facts)},
        )
