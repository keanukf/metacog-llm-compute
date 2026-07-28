"""
Calibration and discrimination metrics for RQ1 (how well the metacognitive signals track step
correctness): ECE, MCE, Brier, reliability-diagram data, AUROC/AUPRC, and the per-step-position
breakdown feeding the H3 temporal-degradation analysis.

Dependency-free by design (pure-Python, no numpy/scipy) so the metrics are auditable and stable.
``compute_auroc`` is the discrimination workhorse for H1 (TLE score = negated mean entropy, since
lower entropy should mean more confident/correct). The ``correctness_policy`` switch
(optimal_only vs legal_or_optimal) is the confirmatory-vs-sensitivity label definition from §5.8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

CorrectnessPolicy = Literal["optimal_only", "legal_or_optimal"]


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


def compute_mce(
    predictions: Sequence[float],
    correctness: Sequence[float | int],
    n_bins: int = 10,
) -> float:
    """
    Maximum Calibration Error: max over bins of |accuracy(bin) - confidence(bin)|.
    """
    predictions = list(predictions)
    correctness = list(correctness)
    if not predictions or len(predictions) != len(correctness):
        return 0.0
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    mce = 0.0
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
        mce = max(mce, abs(acc - conf))
    return float(mce)


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
        mask = (
            [low <= p < high for p in predictions]
            if i < n_bins - 1
            else [low <= p <= 1.0 for p in predictions]
        )
        count = sum(mask)
        if count == 0:
            continue
        bin_centers.append((low + high) / 2)
        mean_conf.append(sum(predictions[j] for j in range(len(mask)) if mask[j]) / count)
        mean_acc.append(sum(correctness[j] for j in range(len(mask)) if mask[j]) / count)
    return bin_centers, mean_conf, mean_acc


def compute_auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """
    AUROC via Mann–Whitney U statistic with average ranks for ties.

    Args:
        scores: Continuous scores (higher should indicate positive class).
        labels: 0/1 labels.

    Returns:
        AUROC in [0, 1]. Returns 0.5 if AUROC is undefined (no positives or no negatives).
    """
    xs = list(scores)
    ys = [int(label) for label in labels]
    if len(xs) != len(ys) or not xs:
        return 0.5
    n_pos = sum(1 for y in ys if y == 1)
    n_neg = len(ys) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Average ranks for ties
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    rank = 1
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        # ranks rank..rank+(j-i) inclusive
        avg = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        rank += j - i + 1
        i = j + 1

    rank_sum_pos = sum(ranks[i] for i in range(len(xs)) if ys[i] == 1)
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def compute_auprc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """
    Area under Precision-Recall curve for binary labels (0/1).

    This is a simple, dependency-free implementation using threshold sweeps over sorted scores.
    Returns the step-wise (right-continuous) PR AUC.
    """
    xs = list(scores)
    ys = [int(label) for label in labels]
    if len(xs) != len(ys) or not xs:
        return 0.0
    n_pos = sum(1 for y in ys if y == 1)
    if n_pos == 0:
        return 0.0

    order = sorted(range(len(xs)), key=lambda i: xs[i], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for idx in order:
        if ys[idx] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / max(tp + fp, 1)
        # Integrate precision w.r.t recall (step function)
        area += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return float(area)


def compute_efficiency(success_rate: float, normalized_compute_cost: float) -> float | None:
    """Efficiency score: success_rate / normalized_compute_cost. Returns None if cost is 0.

    Descriptive only; thesis reports Pareto (success and tokens as separate DVs).
    """
    if normalized_compute_cost <= 0:
        return None
    return float(success_rate) / float(normalized_compute_cost)


def vc_to_prob(vc: float) -> float:
    """Map a verbalized confidence in [0,100] into a probability in [0,1]."""
    return max(0.0, min(1.0, float(vc) / 100.0))


def tle_entropy_to_prob(mean_entropy: float) -> float:
    """
    Baseline monotonic mapping from token-level entropy to a probability in [0,1].

    Note: entropy is not a probability; this mapping is only a heuristic baseline.
    For thesis-grade calibration, prefer a fitted calibrator trained on Phase-1 validation.
    """
    return max(0.0, min(1.0, 1.0 - float(mean_entropy)))


@dataclass(frozen=True)
class FittedTLECalibrator:
    """Logistic mapping from raw TLE (mean entropy) to a correctness probability.

    The "fitted calibrator" ``tle_entropy_to_prob``'s own docstring points to -- see
    ``fit_tle_calibrator`` below for how this gets fit. Breaks this module's own
    dependency-free-by-design norm (uses ``statsmodels``), same precedent as
    ``src.analysis.inference.fit_h3_model``.
    """

    intercept: float
    slope: float

    def predict_proba(self, mean_entropy: float) -> float:
        z = self.intercept + self.slope * float(mean_entropy)
        # Numerically stable logistic: split on the sign of z to avoid overflow in exp() for
        # extreme entropy values.
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)


def fit_tle_calibrator(
    holdout_steps: list[dict[str, Any]],
    *,
    label: str = "y_optimal",
) -> FittedTLECalibrator | dict[str, Any]:
    """
    Logistic regression y ~ tle_mean_entropy, fit on holdout steps.

    Deliberately pooled across runs AND compute stages -- verified verbatim in
    ``chapters/05_methodology.md`` (thesis Ch.5 §5.9): "pooling all holdout steps across runs
    and stages [mitigates variance from the small holdout]". This is a different, simpler,
    separately-preregistered design decision from H3's stage-conditional standardization
    (ADR-006) -- do not carry that pattern over here by analogy.

    Returns an error dict (``{"converged": False, "note": ...}``) on insufficient data,
    single-class label, or non-convergence, mirroring ``fit_h3_model``'s fallback contract so
    callers can branch the same way.
    """
    y: list[int] = []
    x: list[float] = []
    for r in holdout_steps:
        lv = r.get(label)
        tle = r.get("tle_mean_entropy")
        if lv is None or tle is None:
            continue
        y.append(int(lv))
        x.append(float(tle))
    if len(y) < 20 or len(set(y)) < 2:
        return {"converged": False, "note": "insufficient data or single-class label"}
    try:
        import statsmodels.api as sm

        design = sm.add_constant(x)
        model = sm.Logit(y, design)
        res = model.fit(disp=0)
        return FittedTLECalibrator(intercept=float(res.params[0]), slope=float(res.params[1]))
    except Exception as e:
        return {"converged": False, "note": str(e)}


def _label_from_correctness(corr: Any, policy: CorrectnessPolicy) -> float | None:
    if corr is None:
        return None
    if isinstance(corr, str):
        c = corr.strip().lower()
        if policy == "optimal_only":
            if c == "optimal":
                return 1.0
            if c in {"legal", "illegal"}:
                return 0.0
            return None
        if policy == "legal_or_optimal":
            if c in {"optimal", "legal"}:
                return 1.0
            if c == "illegal":
                return 0.0
            return None
        return None
    if isinstance(corr, (int, float)):
        return 1.0 if float(corr) > 0 else 0.0
    return None


def _cohens_d(a: Sequence[float], b: Sequence[float]) -> float | None:
    a = list(a)
    b = list(b)
    if len(a) < 2 or len(b) < 2:
        return None
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = ((va + vb) / 2.0) ** 0.5
    if pooled == 0:
        return None
    return float((ma - mb) / pooled)


def calibration_by_step_position(
    episodes: list[dict[str, Any]],
    signal: str,  # "tle" or "vc"
    n_bins: int = 4,
    correctness_key: str = "correctness",
    correctness_policy: CorrectnessPolicy = "optimal_only",
) -> list[dict[str, Any]]:
    """
    Compute calibration metrics (ECE, Brier) per step-position bin.

    Each episode should contain `steps_detail` with per-step dicts. Older episode schemas are
    supported if callers provide `steps_detail` (loaders will synthesize it).
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    # Determine representative step ranges using max observed episode length.
    lengths = [len(ep.get("steps_detail") or []) for ep in episodes]
    max_len = max(lengths) if lengths else 0
    edges = [int(round(i * max_len / n_bins)) for i in range(n_bins + 1)]

    per_bin_preds: list[list[float]] = [[] for _ in range(n_bins)]
    per_bin_labels: list[list[float]] = [[] for _ in range(n_bins)]

    for ep in episodes:
        steps = ep.get("steps_detail") or []
        if not steps:
            continue
        ep_len = len(steps)
        for sd in steps:
            if not isinstance(sd, dict):
                continue
            idx = sd.get("step_index")
            if isinstance(idx, int):
                pass
            elif idx is None:
                continue
            else:
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    continue
            bin_idx = int((idx / max(ep_len, 1)) * n_bins)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            if bin_idx < 0:
                bin_idx = 0

            # signal value
            if signal == "tle":
                tle = sd.get("tle")
                if not isinstance(tle, dict):
                    continue
                v = tle.get("mean_entropy")
                if not isinstance(v, (int, float)):
                    continue
                pred01 = tle_entropy_to_prob(float(v))
            elif signal == "vc":
                v = sd.get("vc")
                if not isinstance(v, (int, float)):
                    continue
                pred01 = vc_to_prob(float(v))
            else:
                raise ValueError("signal must be 'tle' or 'vc'")

            corr = sd.get(correctness_key)
            correct01 = _label_from_correctness(corr, correctness_policy)
            if correct01 is None:
                continue

            per_bin_preds[bin_idx].append(pred01)
            per_bin_labels[bin_idx].append(correct01)

    out: list[dict[str, Any]] = []
    for i in range(n_bins):
        preds = per_bin_preds[i]
        labs = per_bin_labels[i]
        if preds:
            ece = compute_ece(preds, labs, n_bins=10)
            brier = compute_brier(preds, labs)
        else:
            ece = 0.0
            brier = 0.0
        label = (
            ["early", "mid", "late"][i]
            if n_bins == 3
            else ["early", "mid_early", "mid_late", "late"][i]
            if n_bins == 4
            else f"bin_{i}"
        )
        out.append(
            {
                "bin": label,
                "step_range": (edges[i], max(edges[i], edges[i + 1] - 1)),
                "ece": float(ece),
                "brier": float(brier),
                "n_steps": int(len(preds)),
            }
        )
    return out


def signal_discrimination_report(
    episodes: list[dict[str, Any]],
    signal: str,
    *,
    collapse_policy: CorrectnessPolicy = "optimal_only",
) -> dict[str, Any]:
    """
    Compute signal discrimination on step-level correctness.

    Returns AUROC, mean signal for correct vs incorrect steps, and Cohen's d.
    """
    scores: list[float] = []
    labels: list[int] = []
    sig_correct: list[float] = []
    sig_incorrect: list[float] = []
    for ep in episodes:
        for sd in ep.get("steps_detail") or []:
            if not isinstance(sd, dict):
                continue
            corr = sd.get("correctness")
            y01 = _label_from_correctness(corr, collapse_policy)
            if y01 is None:
                continue
            if signal == "tle":
                tle = sd.get("tle")
                if not isinstance(tle, dict):
                    continue
                v = tle.get("mean_entropy")
                if not isinstance(v, (int, float)):
                    continue
                raw = float(v)
                # AUROC: higher score = more likely optimal (lower entropy = more confident).
                s = -raw
            elif signal == "vc":
                v = sd.get("vc")
                if not isinstance(v, (int, float)):
                    continue
                raw = float(v)
                s = raw
            else:
                raise ValueError("signal must be 'tle' or 'vc'")
            scores.append(s)
            y = 1 if y01 >= 0.5 else 0
            labels.append(y)
            (sig_correct if y == 1 else sig_incorrect).append(raw)

    auroc = compute_auroc(scores, labels)
    mean_correct = (sum(sig_correct) / len(sig_correct)) if sig_correct else 0.0
    mean_incorrect = (sum(sig_incorrect) / len(sig_incorrect)) if sig_incorrect else 0.0
    d = _cohens_d(sig_correct, sig_incorrect)
    return {
        "signal": signal,
        "auroc": float(auroc),
        "mean_signal_correct": float(mean_correct),
        "mean_signal_incorrect": float(mean_incorrect),
        "cohens_d": d,
        "n_steps": int(len(scores)),
        "collapse_policy": collapse_policy,
    }


def compare_signal_calibration(
    episodes: list[dict[str, Any]],
    *,
    signals: tuple[str, ...] = ("tle", "vc"),
    collapse_policies: tuple[CorrectnessPolicy, ...] = ("optimal_only", "legal_or_optimal"),
) -> dict[str, Any]:
    """Run confirmatory + sensitivity calibration reports for each signal."""
    out: dict[str, Any] = {}
    for pol in collapse_policies:
        out[pol] = {
            sig: signal_discrimination_report(episodes, sig, collapse_policy=pol) for sig in signals
        }
    return out


def compute_strategy_efficiency(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Descriptive group summaries only — not a confirmatory decision metric.

    Groups by strategy (Phase 2) or compute_stage (Phase 1).
    Thesis primary reporting uses separate success and token DVs (Pareto).
    """
    # Detect group key
    group_key = "strategy" if any("strategy" in ep for ep in episodes) else "compute_stage"
    groups: dict[str, list[dict[str, Any]]] = {}
    for ep in episodes:
        g = str(ep.get(group_key, "unknown"))
        groups.setdefault(g, []).append(ep)

    rows: list[dict[str, Any]] = []
    for g, eps in groups.items():
        n = len(eps)
        if n == 0:
            continue
        succ = sum(1 for e in eps if e.get("task_success"))
        success_rate = succ / n
        costs = [float(e.get("normalized_compute_cost") or 0.0) for e in eps]
        avg_cost = sum(costs) / n if n else 0.0
        eff = compute_efficiency(success_rate, avg_cost)
        rows.append(
            {
                group_key: g,
                "episodes": n,
                "success_rate": float(success_rate),
                "avg_normalized_compute_cost": float(avg_cost),
                "efficiency_score": eff,
            }
        )
    return sorted(rows, key=lambda r: r.get(group_key, ""))
