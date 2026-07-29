"""Full per-variable descriptive statistics / APA-7 table rendering (src.analysis.descriptive_stats).

Exercises describe_values against hand-computable synthetic data, missingness accounting relative
to an explicit total, the per-domain/per-stage codebook shape, and that the rendered APA-style
markdown contains the expected table titles and numeric cells.
"""

from __future__ import annotations

from src.analysis.descriptive_stats import (
    compute_sample_composition,
    compute_signal_correlation,
    compute_variable_codebook,
    describe_values,
    describe_variable,
    render_apa_codebook_markdown,
)


def test_describe_values_matches_hand_computed_stats():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    d = describe_values(vals)
    assert d["n"] == 5
    assert d["mean"] == 3.0
    assert d["median"] == 3.0
    assert d["min"] == 1.0
    assert d["max"] == 5.0
    assert d["q1"] < d["median"] < d["q3"]
    # Symmetric distribution -> skewness ~ 0.
    assert abs(d["skewness"]) < 1e-9


def test_describe_values_empty_input():
    d = describe_values([])
    assert d["n"] == 0
    assert d["mean"] is None
    assert d["skewness"] is None


def test_describe_variable_reports_missingness_relative_to_total():
    rows = [{"x": 1.0}, {"x": 2.0}, {"x": None}, {}]
    d = describe_variable(rows, "x")
    assert d["n"] == 2
    assert d["n_missing"] == 2
    assert d["missing_rate"] == 0.5


def test_describe_variable_respects_explicit_total():
    rows = [{"x": 1.0}, {"x": 2.0}]
    d = describe_variable(rows, "x", total=10)
    assert d["n"] == 2
    assert d["n_missing"] == 8
    assert d["missing_rate"] == 0.8


def _fixture_steps_and_episodes():
    steps = []
    for dom in ("tower_of_hanoi", "textworld"):
        for stage in ("C0", "C1", "C2"):
            for i in range(10):
                steps.append(
                    {
                        "domain": dom,
                        "compute_stage": stage,
                        "tle_mean_entropy": 0.1 * i,
                        "vc": 100 - 10 * i,
                        "position_norm": i / 9.0,
                        "y_optimal": 1 if i % 2 == 0 else 0,
                    }
                )
    episodes = [
        {
            "domain": dom,
            "compute_stage": stage,
            "holdout": i < 1,
            "task_success": i % 2 == 0,
            "episode_length_steps": 10 + i,
            "normalized_compute_cost": 0.5 + 0.01 * i,
        }
        for dom in ("tower_of_hanoi", "textworld")
        for stage in ("C0", "C1", "C2")
        for i in range(5)
    ]
    return steps, episodes


def test_compute_variable_codebook_shape():
    steps, episodes = _fixture_steps_and_episodes()
    codebook = compute_variable_codebook(steps, episodes)

    for dom in ("tower_of_hanoi", "textworld"):
        assert dom in codebook["step_level"]
        assert set(codebook["step_level"][dom].keys()) == {
            "tle_mean_entropy",
            "vc",
            "position_norm",
            "y_optimal",
        }
        assert codebook["step_level"][dom]["tle_mean_entropy"]["n"] == 30  # 3 stages x 10
        assert codebook["step_level"][dom]["y_optimal"]["rate"] == 0.5

        assert set(codebook["step_level_by_stage"][dom].keys()) == {"C0", "C1", "C2"}
        for stage_report in codebook["step_level_by_stage"][dom].values():
            assert stage_report["tle_mean_entropy"]["n"] == 10
            assert stage_report["y_optimal"]["rate"] == 0.5

        ep = codebook["episode_level"][dom]
        assert ep["n_episodes"] == 15  # 3 stages x 5
        assert ep["task_success_rate"] == 0.6  # 3 of 5 (i=0,2,4) succeed, per stage, pooled

        corr = codebook["signal_correlation"][dom]
        # TLE increases with i, VC decreases with i -> perfect negative monotonic relationship.
        assert corr["pearson_r"] < -0.99
        assert corr["spearman_rho"] < -0.99

    composition = codebook["sample_composition"]
    assert len(composition) == 2 * 3 * 2  # domain x stage x holdout(True/False)
    holdout_row = next(
        r for r in composition if r["domain"] == "tower_of_hanoi" and r["compute_stage"] == "C0" and r["holdout"]
    )
    assert holdout_row["n_episodes"] == 1  # only i=0 has holdout=True


def test_compute_signal_correlation_insufficient_data_returns_none():
    c = compute_signal_correlation([{"tle_mean_entropy": 0.1, "vc": 50}])
    assert c["n"] == 1
    assert c["pearson_r"] is None
    assert c["spearman_rho"] is None


def test_compute_sample_composition_counts_design_cells():
    episodes = [
        {"domain": "tower_of_hanoi", "compute_stage": "C0", "holdout": False} for _ in range(3)
    ] + [{"domain": "tower_of_hanoi", "compute_stage": "C0", "holdout": True}]
    rows = compute_sample_composition(episodes)
    assert {"domain": "tower_of_hanoi", "compute_stage": "C0", "holdout": False, "n_episodes": 3} in rows
    assert {"domain": "tower_of_hanoi", "compute_stage": "C0", "holdout": True, "n_episodes": 1} in rows


def test_render_apa_codebook_markdown_contains_expected_tables():
    steps, episodes = _fixture_steps_and_episodes()
    codebook = compute_variable_codebook(steps, episodes)
    md = render_apa_codebook_markdown(codebook)

    for i in range(1, 8):
        assert f"*Table {i}*" in md
    assert "Variable roles and measurement scales" in md
    assert "Sample composition" in md
    assert "step-level signals, by domain" in md
    assert "by domain and compute stage" in md
    assert "correctness rate (y_optimal)" in md
    assert "episode-level variables, by domain" in md
    assert "TLE-VC association" in md
    assert "tower_of_hanoi" in md and "textworld" in md
    assert "*Note.*" in md
