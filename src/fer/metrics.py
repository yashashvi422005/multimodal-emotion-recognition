"""Scoring utilities shared across Phase 4 (occlusion) and Phase 5 (fusion).

Official abstention rule (face branch, or any branch that can decline to
vote): an abstained clip is a miss for its true class (false negative in
that class's F1), is NOT a prediction of any class (no false positive for
whatever a tie-break would otherwise pick), and still counts in the
accuracy denominator. This differs from both naive alternatives — scoring
the tie-broken class as a real prediction (inflates/deflates whichever
class wins the tie-break), or dropping abstained clips entirely (shrinks
the denominator, hiding how often a condition causes the branch to fail
outright).
"""

from __future__ import annotations

import numpy as np


def score_with_abstention(y_true, y_pred, abstained, class_order: list[str]) -> tuple[float, float, dict[str, float]]:
    """y_pred for an abstained row is ignored — it contributes a false
    negative to its true class and nothing else. Returns
    (macro_f1, accuracy, per_class_f1)."""
    tp = {c: 0 for c in class_order}
    fp = {c: 0 for c in class_order}
    fn = {c: 0 for c in class_order}
    n_correct = 0
    n_total = len(y_true)

    for yt, yp, ab in zip(y_true, y_pred, abstained):
        if ab:
            fn[yt] += 1
            continue
        if yt == yp:
            tp[yt] += 1
            n_correct += 1
        else:
            fn[yt] += 1
            fp[yp] += 1

    per_class_f1 = {}
    for c in class_order:
        denom = 2 * tp[c] + fp[c] + fn[c]
        per_class_f1[c] = (2 * tp[c] / denom) if denom > 0 else 0.0

    macro_f1 = float(np.mean(list(per_class_f1.values())))
    accuracy = n_correct / n_total if n_total > 0 else float("nan")
    return macro_f1, accuracy, per_class_f1


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Standard binned ECE: mean absolute gap between confidence and
    accuracy within equal-width confidence bins, weighted by bin size."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == bin_edges[-1]:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)
