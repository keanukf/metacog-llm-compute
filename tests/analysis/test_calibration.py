"""``fit_tle_calibrator``/``FittedTLECalibrator`` (``src/analysis/calibration.py``).

No test file existed for this module before 2026-07-28 (Phase 1 analysis pipeline, Stage 3).
Covers the new logistic TLE->probability calibrator H1b depends on: fits a real negative slope
on monotonic synthetic data (higher entropy -> lower correctness), degrades gracefully (never
raises) on insufficient or single-class data, and ``predict_proba`` stays numerically stable and
bounded in [0,1] across an extreme input range.
"""

from __future__ import annotations

import random

import pytest

from src.analysis.calibration import FittedTLECalibrator, fit_tle_calibrator


def _monotonic_holdout_steps(n: int, *, seed: int = 1) -> list[dict]:
    """Higher entropy -> lower P(correct): a clean logistic-shaped signal."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        tle = rng.uniform(0.0, 1.0)
        p_correct = 1.0 - tle
        y = 1 if rng.random() < p_correct else 0
        rows.append({"tle_mean_entropy": tle, "y_optimal": y})
    return rows


def test_fit_tle_calibrator_recovers_negative_slope_on_monotonic_data():
    rows = _monotonic_holdout_steps(300)
    result = fit_tle_calibrator(rows)
    assert isinstance(result, FittedTLECalibrator)
    assert result.slope < 0  # higher entropy -> lower predicted probability of correctness


def test_predict_proba_is_bounded_and_monotonic():
    calibrator = FittedTLECalibrator(intercept=1.0, slope=-2.0)
    p_low = calibrator.predict_proba(0.0)
    p_high = calibrator.predict_proba(1.0)
    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0
    assert p_low > p_high  # slope < 0 => higher entropy => lower probability


def test_predict_proba_numerically_stable_at_extreme_inputs():
    calibrator = FittedTLECalibrator(intercept=0.0, slope=-50.0)
    assert 0.0 <= calibrator.predict_proba(100.0) <= 1.0
    assert 0.0 <= calibrator.predict_proba(-100.0) <= 1.0


def test_predict_proba_matches_scipy_expit_directly():
    """Regression for the 2026-08-03 sklearn/scipy swap: predict_proba must be exactly the
    logistic sigmoid of (intercept + slope*x), not an approximation of it."""
    from scipy.special import expit

    calibrator = FittedTLECalibrator(intercept=0.7, slope=-3.2)
    for x in (-1000.0, -5.0, -0.001, 0.0, 0.3, 5.0, 1000.0):
        expected = float(expit(calibrator.intercept + calibrator.slope * x))
        assert calibrator.predict_proba(x) == pytest.approx(expected, abs=1e-12)


def test_fit_tle_calibrator_fails_gracefully_on_insufficient_rows():
    result = fit_tle_calibrator([{"tle_mean_entropy": 0.1, "y_optimal": 1}] * 5)
    assert isinstance(result, dict)
    assert result["converged"] is False


def test_fit_tle_calibrator_fails_gracefully_on_single_class_label():
    rows = [{"tle_mean_entropy": i / 100.0, "y_optimal": 1} for i in range(50)]
    result = fit_tle_calibrator(rows)
    assert isinstance(result, dict)
    assert result["converged"] is False


def test_fit_tle_calibrator_skips_rows_missing_fields():
    rows = _monotonic_holdout_steps(300)
    rows += [{"tle_mean_entropy": None, "y_optimal": 1}] * 50
    rows += [{"y_optimal": 0}] * 50  # missing tle_mean_entropy entirely
    result = fit_tle_calibrator(rows)
    assert isinstance(result, FittedTLECalibrator)  # the well-formed 300 rows are still enough
