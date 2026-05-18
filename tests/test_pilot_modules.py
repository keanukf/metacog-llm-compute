from __future__ import annotations

from pathlib import Path

from src.pilot.artifacts import load_json_optional, save_json
from src.pilot.steps import (
    episode_vc_tle_rates,
    prepare_feasibility_inputs,
    run_mock_inference_speed_benchmark,
)


def test_artifact_save_and_load_roundtrip(tmp_path: Path) -> None:
    payload = {"ok": True, "value": 7}
    out = save_json(tmp_path, "pilot_test", payload)
    loaded = load_json_optional(out)
    assert loaded == payload


def test_prepare_feasibility_inputs_uses_disk_fallbacks(tmp_path: Path) -> None:
    save_json(tmp_path, "pilot_test1_inference", {"tokens_per_sec": 12.3})
    save_json(tmp_path, "pilot_test2_tle", {"tle_ok": True})
    save_json(tmp_path, "pilot_test3_vc", {"vc_ok": True})
    save_json(tmp_path, "pilot_test5_toh", {"parse_rate": 0.8})
    save_json(tmp_path, "pilot_sanity", {"ok": True})
    save_json(tmp_path, "ep_textworld_0", {"vc_per_step": [10], "tle_per_step": [{"mean_entropy": 0.2}]})
    save_json(tmp_path, "ep_tower_of_hanoi_0", {"vc_per_step": [20], "tle_per_step": [{"mean_entropy": 0.1}]})

    t1, t2, t3, eps, th, san, toh_eps = prepare_feasibility_inputs(
        tmp_path,
        test1={},
        test2={},
        test3={},
        episodes=[],
        toh={},
        sanity=None,
    )
    assert t1["tokens_per_sec"] == 12.3
    assert t2["tle_ok"] is True
    assert t3["vc_ok"] is True
    assert th["parse_rate"] == 0.8
    assert san and san["ok"] is True
    assert len(eps) == 1
    assert len(toh_eps) == 1


def test_episode_vc_tle_rates_counts_non_null_values() -> None:
    vc_rate, tle_rate = episode_vc_tle_rates(
        [{"vc_per_step": [10, None], "tle_per_step": [{"mean_entropy": 0.1}, None]}],
        [{"vc_per_step": [None], "tle_per_step": [None]}],
    )
    assert vc_rate == 1 / 3
    assert tle_rate == 1 / 3


def test_mock_benchmark_returns_expected_keys() -> None:
    out = run_mock_inference_speed_benchmark(5, tokens_per_call=10)
    assert out["num_prompts"] == 5
    assert out["total_tokens"] == 50
    assert out["tokens_per_sec"] > 0
    assert "latency_mean" in out
    assert "latency_std" in out
