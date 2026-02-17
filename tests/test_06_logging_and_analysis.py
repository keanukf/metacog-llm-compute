"""
Pilot Test 6 — Logging & Download.
(1) Round-trip: write episode via logging_utils, read back, assert equality.
(2) ECE: 15 synthetic data points -> compute_ece -> result in [0, 1].
"""
from __future__ import annotations

import json

import pytest

from src.analysis.calibration import compute_ece
from src.utils.logging_utils import log_episode


def test_log_episode_round_trip(sample_episode_data, temp_results_dir):
    ep_id = sample_episode_data["episode_id"]
    path = log_episode(ep_id, sample_episode_data, temp_results_dir)
    assert path.exists()
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == sample_episode_data


def test_compute_ece_returns_number():
    predictions = [0.2, 0.5, 0.8, 0.3, 0.9] * 3
    correctness = [0, 1, 1, 0, 1] * 3
    ece = compute_ece(predictions, correctness, n_bins=5)
    assert isinstance(ece, (int, float))
    assert 0 <= ece <= 1


def test_compute_ece_on_15_points():
    predictions = [0.1 * (i % 10) for i in range(15)]
    correctness = [1 if p > 0.5 else 0 for p in predictions]
    ece = compute_ece(predictions, correctness, n_bins=5)
    assert 0 <= ece <= 1
