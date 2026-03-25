"""
Calibration metrics: ECE (Expected Calibration Error), Brier score, reliability diagrams.
"""
from __future__ import annotations

from typing import Any, Sequence


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
    ys = [int(l) for l in labels]
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
        rank += (j - i + 1)
        i = j + 1

    rank_sum_pos = sum(ranks[i] for i in range(len(xs)) if ys[i] == 1)
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def compute_efficiency(success_rate: float, normalized_compute_cost: float) -> float | None:
    """Efficiency score: success_rate / normalized_compute_cost. Returns None if cost is 0."""
    if normalized_compute_cost <= 0:
        return None
    return float(success_rate) / float(normalized_compute_cost)


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
    return (ma - mb) / pooled


def calibration_by_step_position(
    episodes: list[dict[str, Any]],
    signal: str,  # "tle" or "vc"
    n_bins: int = 4,
    correctness_key: str = "correctness",
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
            if not isinstance(idx, int):
                try:
                    idx = int(idx)
                except Exception:
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
                pred = float(v)
                # TLE isn't a probability; for calibration we map higher entropy -> lower confidence
                # by a simple monotonic transform into [0,1] using a soft clamp.
                # This keeps the function usable without needing per-domain calibration mapping here.
                pred01 = max(0.0, min(1.0, 1.0 - pred))
            elif signal == "vc":
                v = sd.get("vc")
                if not isinstance(v, (int, float)):
                    continue
                pred01 = max(0.0, min(1.0, float(v) / 100.0))
            else:
                raise ValueError("signal must be 'tle' or 'vc'")

            corr = sd.get(correctness_key)
            if corr is None:
                continue
            # Default: treat any legal (or optimal) move as correct; illegal as incorrect.
            correct01 = 0.0
            if isinstance(corr, str):
                if corr.lower() in {"optimal", "legal"}:
                    correct01 = 1.0
                elif corr.lower() == "illegal":
                    correct01 = 0.0
                else:
                    continue
            elif isinstance(corr, (int, float)):
                correct01 = 1.0 if float(corr) > 0 else 0.0
            else:
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
            ["early", "mid", "late"][i] if n_bins == 3 else ["early", "mid_early", "mid_late", "late"][i] if n_bins == 4 else f"bin_{i}"
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


def signal_discrimination_report(episodes: list[dict[str, Any]], signal: str) -> dict[str, Any]:
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
            if not isinstance(corr, str):
                continue
            y = 1 if corr.lower() in {"optimal", "legal"} else 0 if corr.lower() == "illegal" else None
            if y is None:
                continue
            if signal == "tle":
                tle = sd.get("tle")
                if not isinstance(tle, dict):
                    continue
                v = tle.get("mean_entropy")
                if not isinstance(v, (int, float)):
                    continue
                s = float(v)
            elif signal == "vc":
                v = sd.get("vc")
                if not isinstance(v, (int, float)):
                    continue
                s = float(v)
            else:
                raise ValueError("signal must be 'tle' or 'vc'")
            scores.append(s)
            labels.append(y)
            (sig_correct if y == 1 else sig_incorrect).append(s)

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
    }


def compute_strategy_efficiency(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group episodes by strategy (Phase 2) or compute_stage (Phase 1) and compute
    success_rate, avg normalized compute, and efficiency_score.
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
