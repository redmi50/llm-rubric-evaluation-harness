"""Inter-grader agreement, used for calibration sessions.

Cohen's kappa measures agreement between two graders on the same items, corrected
for the agreement that would be expected by chance alone. A kappa near 1 means the
two graders are interchangeable. A kappa near 0 means they agree no better than
chance, which is the signal that a rubric needs its level anchors rewritten.
"""

from __future__ import annotations

DEGENERATE_TOLERANCE = 1e-12


class AgreementError(ValueError):
    """Raised when the inputs to an agreement calculation are inconsistent."""


def percent_agreement(labels_a, labels_b) -> float:
    """The proportion of items where the two graders assigned the same label."""
    _validate(labels_a, labels_b)
    matches = sum(1 for left, right in zip(labels_a, labels_b) if left == right)
    return matches / len(labels_a)


def cohen_kappa(labels_a, labels_b) -> float:
    """Cohen's kappa for two label sequences.

    The degenerate case is defined explicitly. When both graders used a single
    label each, so expected agreement is 1.0, the function returns 1.0 if they
    agreed on every item and 0.0 otherwise. The alternative convention of
    returning NaN is not used here because a NaN silently propagates into a
    report and reads as a passing result.
    """
    _validate(labels_a, labels_b)

    total = len(labels_a)
    observed = sum(1 for left, right in zip(labels_a, labels_b) if left == right) / total

    categories = sorted({*labels_a, *labels_b}, key=repr)
    expected = 0.0
    for category in categories:
        share_a = sum(1 for label in labels_a if label == category) / total
        share_b = sum(1 for label in labels_b if label == category) / total
        expected += share_a * share_b

    if abs(expected - 1.0) <= DEGENERATE_TOLERANCE:
        return 1.0 if abs(observed - 1.0) <= DEGENERATE_TOLERANCE else 0.0

    return (observed - expected) / (1.0 - expected)


def agreement_matrix(label_sets: dict[str, list]) -> dict[str, dict[str, float]]:
    """Pairwise kappa between every pair of named graders.

    Each value in ``label_sets`` is the full label sequence from one grader, in
    the same item order.
    """
    if len(label_sets) < 2:
        raise AgreementError("At least two graders are required to compute agreement.")

    names = sorted(label_sets)
    lengths = {name: len(labels) for name, labels in label_sets.items()}
    if len(set(lengths.values())) != 1:
        raise AgreementError(
            f"Every grader must label the same number of items, got {lengths}."
        )

    matrix: dict[str, dict[str, float]] = {}
    for left in names:
        matrix[left] = {}
        for right in names:
            if left == right:
                matrix[left][right] = 1.0
            else:
                matrix[left][right] = round(cohen_kappa(label_sets[left], label_sets[right]), 4)
    return matrix


def overall_agreement(label_sets: dict[str, list]) -> float:
    """Mean pairwise kappa across every distinct pair of graders."""
    matrix = agreement_matrix(label_sets)
    names = sorted(matrix)
    values = [
        matrix[left][right]
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    ]
    return sum(values) / len(values) if values else 1.0


def _validate(labels_a, labels_b) -> None:
    if len(labels_a) != len(labels_b):
        raise AgreementError(
            f"Both graders must label the same number of items, "
            f"got {len(labels_a)} and {len(labels_b)}."
        )
    if not len(labels_a):
        raise AgreementError("Cannot compute agreement on an empty label set.")
