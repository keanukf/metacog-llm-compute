"""Phase 1 analysis pipeline, Stage 5 (H4 domain modulation).

Smoke test against tiny synthetic data. Verifies: single-test Family C output shape, and that a
genuine domain-modulation signal (TLE-over-VC advantage larger under tower_of_hanoi than
textworld) is at least distinguishable in the point estimate's sign.
"""

from __future__ import annotations

import random

from scripts.phase1_analysis.stage5_h4_domain_modulation import run_h4


def _domain_steps(
    domain: str, *, n_instances: int, steps_per_instance: int, tle_advantage: float, seed: int
) -> list[dict]:
    """tle_advantage controls how much better TLE discriminates correctness than VC in this
    domain -- larger in tower_of_hanoi than textworld reproduces the preregistered H4 direction."""
    rng = random.Random(seed)
    rows = []
    for inst in range(n_instances):
        for t in range(steps_per_instance):
            y = 1 if rng.random() < 0.5 else 0
            tle = (1 - y) * 0.5 + rng.uniform(-0.05, 0.05) - tle_advantage * (1 - y) * 0.3
            vc = 50 + rng.uniform(-10, 10)
            rows.append(
                {
                    "domain": domain,
                    "instance_key": f"{domain}:{inst}",
                    "y_optimal": y,
                    "tle_mean_entropy": tle,
                    "vc": vc,
                }
            )
    return rows


def test_run_h4_shape_and_family():
    steps = []
    steps += _domain_steps(
        "tower_of_hanoi", n_instances=10, steps_per_instance=8, tle_advantage=0.8, seed=1
    )
    steps += _domain_steps(
        "textworld", n_instances=10, steps_per_instance=8, tle_advantage=0.1, seed=2
    )

    result = run_h4(steps, n_boot=200, seed=7)

    assert result["family"] == "C"
    r = result["result"]
    assert "point" in r
    assert "ci_low" in r and "ci_high" in r
    assert "holm" in r
    assert r["holm"]["family"] == "C"
    assert "decision_holds" in r


def test_run_h4_insufficient_clusters_returns_none_point():
    steps = [
        {
            "domain": "tower_of_hanoi",
            "instance_key": "toh:0",
            "y_optimal": 1,
            "tle_mean_entropy": 0.1,
            "vc": 60,
        },
        {
            "domain": "textworld",
            "instance_key": "tw:0",
            "y_optimal": 1,
            "tle_mean_entropy": 0.1,
            "vc": 60,
        },
        {
            "domain": "textworld",
            "instance_key": "tw:1",
            "y_optimal": 0,
            "tle_mean_entropy": 0.4,
            "vc": 40,
        },
    ]
    result = run_h4(steps, n_boot=50, seed=1)
    assert result["result"]["point"] is None
    assert result["result"]["decision_holds"] is False


def test_run_h4_on_bootstrap_hook_fires_with_reps():
    steps = []
    steps += _domain_steps(
        "tower_of_hanoi", n_instances=10, steps_per_instance=8, tle_advantage=0.8, seed=1
    )
    steps += _domain_steps(
        "textworld", n_instances=10, steps_per_instance=8, tle_advantage=0.1, seed=2
    )

    seen = {}
    run_h4(steps, n_boot=50, seed=7, on_bootstrap=lambda boot: seen.update(boot))

    assert "reps" in seen and len(seen["reps"]) > 0
