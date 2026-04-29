"""
Pilot Test 6 — Logging & Download.
(1) Round-trip: write episode via logging_utils, read back, assert equality.
(2) ECE: 15 synthetic data points -> compute_ece -> result in [0, 1].
"""
from __future__ import annotations

import json

import pytest

from src.analysis.calibration import compute_ece
from src.utils.logging_utils import (
    compact_episode_for_storage,
    log_episode,
    write_logprob_distribution_artifacts,
    write_vc_distribution_artifacts,
)


def test_compact_episode_drops_verbose_keys():
    d = {
        "episode_id": "ep_x",
        "steps_detail": [{"step_index": 0}],
        "vc_detail_per_step": [{"vc_value": 1.0}],
        "logprob_raw_per_step": [[{"logprob": 0.0}]],
        "tle_per_step": [{"mean_entropy": 0.1}],
    }
    c = compact_episode_for_storage(d)
    assert "steps_detail" not in c and "vc_detail_per_step" not in c and "logprob_raw_per_step" not in c
    assert c["tle_per_step"] == [{"mean_entropy": 0.1}]


def test_log_episode_round_trip(sample_episode_data, temp_results_dir):
    ep_id = sample_episode_data["episode_id"]
    path = log_episode(ep_id, sample_episode_data, temp_results_dir, compact=False)
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


def test_write_vc_distribution_artifacts_json(temp_results_dir):
    from src.signals.verbalized_confidence import extract_vc_from_followup

    detail = extract_vc_from_followup("p", "90", [{"token": "9", "logprob": -0.1}])
    paths = write_vc_distribution_artifacts(
        "ep_test_vc_0",
        [detail, None],
        temp_results_dir,
        export_format="json",
        vc_subdir="vc",
    )
    assert len(paths) == 1
    assert paths[0].name == "ep_test_vc_0_vc.json"
    assert paths[0].parent.name == "vc"
    import json

    with open(paths[0]) as f:
        body = json.load(f)
    assert body["steps"][0]["vc_record"]["vc_value"] == 90.0
    assert body["steps"][1]["vc_record"] is None


def test_write_logprob_distribution_artifacts_schema_v2_for_c2_multi_sample(temp_results_dir):
    # Step 0: single-sample (schema v1 compatible)
    lp0 = [{"token": "a", "logprob": -0.1}]
    # Step 1: multi-sample (C2-like) - two samples
    lp1 = [
        [{"token": "b", "logprob": -0.2}],
        [{"token": "c", "logprob": -0.3}],
    ]
    paths = write_logprob_distribution_artifacts(
        "ep_test_lp_c2",
        [lp0, lp1],
        temp_results_dir,
        export_format="json",
        logprob_subdir="logprobs",
    )
    assert len(paths) == 1
    assert paths[0].name == "ep_test_lp_c2_logprobs.json"
    with open(paths[0]) as f:
        body = json.load(f)
    assert body["schema_version"] == 2
    assert body["steps"][0]["logprob_tokens"][0]["token"] == "a"
    assert body["steps"][1]["samples"][0]["sample_index"] == 0
    assert body["steps"][1]["samples"][1]["sample_index"] == 1
