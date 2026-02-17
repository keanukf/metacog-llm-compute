"""
Calibration metrics: ECE (Expected Calibration Error), Brier score, reliability diagrams.
"""
from __future__ import annotations

from typing import Sequence


def compute_ece(
    predictions: Sequence[float],
    correctness: Sequence[float | int],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error: bin predictions, compute weighted average of
    |accuracy(bin) - confidence(bin)|.

    Args:
        predictions: Confidence or probability in [0, 1] per sample.
        correctness: 0/1 or float correctness per sample.
        n_bins: Number of bins for calibration.

    Returns:
        ECE in [0, 1].
    """
    predictions = list(predictions)
    correctness = list(correctness)
    if not predictions or len(predictions) != len(correctness):
        return 0.0
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    total = len(predictions)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = [low <= p < high for p in predictions]
        if i == n_bins - 1:
            mask = [low <= p <= 1.0 for p in predictions]
        count = sum(mask)
        if count == 0:
            continue
        acc = sum(correctness[j] for j in range(len(mask)) if mask[j]) / count
        conf = sum(predictions[j] for j in range(len(mask)) if mask[j]) / count
        ece += (count / total) * abs(acc - conf)
    return ece


def compute_brier(
    predictions: Sequence[float],
    correctness: Sequence[float | int],
) -> float:
    """
    Brier score: mean squared error between predicted probability and outcome.

    Args:
        predictions: Predicted probabilities in [0, 1].
        correctness: 0/1 outcomes.

    Returns:
        Brier score (lower is better).
    """
    predictions = list(predictions)
    correctness = list(correctness)
    if not predictions or len(predictions) != len(correctness):
        return 0.0
    return sum((p - c) ** 2 for p, c in zip(predictions, correctness)) / len(predictions)


def reliability_diagram_data(
    predictions: Sequence[float],
    correctness: Sequence[float | int],
    n_bins: int = 10,
) -> tuple[list[float], list[float], list[float]]:
    """
    Data for a reliability diagram: bin centers, mean predicted confidence, mean accuracy per bin.

    Returns:
        (bin_centers, mean_confidence_per_bin, mean_accuracy_per_bin).
    """
    predictions = list(predictions)
    correctness = list(correctness)
    if not predictions:
        return [], [], []
    bin_centers = []
    mean_conf = []
    mean_acc = []
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        mask = [low <= p < high for p in predictions] if i < n_bins - 1 else [low <= p <= 1.0 for p in predictions]
        count = sum(mask)
        if count == 0:
            continue
        bin_centers.append((low + high) / 2)
        mean_conf.append(sum(predictions[j] for j in range(len(mask)) if mask[j]) / count)
        mean_acc.append(sum(correctness[j] for j in range(len(mask)) if mask[j]) / count)
    return bin_centers, mean_conf, mean_acc
