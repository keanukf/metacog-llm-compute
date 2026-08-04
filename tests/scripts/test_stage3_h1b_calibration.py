"""Phase 1 analysis pipeline, Stage 3 (H1b calibration).

Smoke test against tiny synthetic data. Verifies: the calibrator is fit on holdout steps only,
the confirmatory DeltaBrier is evaluated on non-holdout steps only (the two must not overlap --
that would leak the calibration target into its own evaluation set), and a domain whose
calibrator fails to converge is reported rather than silently crashing the whole stage.
"""

from __future__ import annotations

import random

from scripts.phase1_analysis.stage3_h1b_calibration import DOMAINS, run_h1b


def _rows(
    domain: str, *, n: int, holdout_fraction: float, tle_discriminates: bool, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    n_holdout_instances = max(1, int(10 * holdout_fraction))
    for i in range(n):
        instance = i % 10
        holdout = instance < n_holdout_instances
        tle = rng.uniform(0.0, 1.0)
        p_correct = (1.0 - tle) if tle_discriminates else 0.5
        y = 1 if rng.random() < p_correct else 0
        vc = rng.uniform(0, 100)
        rows.append(
            {
                "domain": domain,
                "instance_key": f"{domain}:{instance}",
                "holdout": holdout,
                "y_optimal": y,
                "tle_mean_entropy": tle,
                "vc": vc,
            }
        )
    return rows


def test_run_h1b_uses_disjoint_holdout_and_evaluation_sets():
    steps = []
    for dom in DOMAINS:
        steps += _rows(
            dom, n=400, holdout_fraction=0.1, tle_discriminates=True, seed=hash(dom) % 1000
        )

    result = run_h1b(steps, n_boot=200, seed=1)

    assert result["family"] == "D"
    for dom in DOMAINS:
        d = result["by_domain"][dom]
        assert d["calibrator_converged"] is True
        assert d["n_holdout_steps"] > 0
        assert "point" in d and "ci_low" in d and "ci_high" in d
        assert "decision_holds" in d
        assert "holm" in d


def test_run_h1b_reports_nonconverging_domain_without_crashing():
    steps = []
    # tower_of_hanoi: fine, discriminating signal.
    steps += _rows("tower_of_hanoi", n=400, holdout_fraction=0.1, tle_discriminates=True, seed=1)
    # textworld: single-class holdout label (all correct) -> calibrator can't fit.
    for i in range(50):
        steps.append(
            {
                "domain": "textworld",
                "instance_key": f"textworld:{i % 5}",
                "holdout": True,
                "y_optimal": 1,
                "tle_mean_entropy": 0.3,
                "vc": 50,
            }
        )

    result = run_h1b(steps, n_boot=100, seed=1)
    assert result["by_domain"]["tower_of_hanoi"]["calibrator_converged"] is True
    assert result["by_domain"]["textworld"]["calibrator_converged"] is False
    assert result["by_domain"]["textworld"]["decision_holds"] is False


def test_run_h1b_hooks_fire_only_for_converged_domains():
    steps = []
    steps += _rows("tower_of_hanoi", n=400, holdout_fraction=0.1, tle_discriminates=True, seed=1)
    for i in range(50):  # textworld: single-class holdout -> calibrator fit fails
        steps.append(
            {
                "domain": "textworld",
                "instance_key": f"textworld:{i % 5}",
                "holdout": True,
                "y_optimal": 1,
                "tle_mean_entropy": 0.3,
                "vc": 50,
            }
        )

    boot_seen: dict[str, dict] = {}
    fit_seen: dict[str, object] = {}
    run_h1b(
        steps,
        n_boot=50,
        seed=1,
        on_bootstrap=lambda dom, boot: boot_seen.__setitem__(dom, boot),
        on_calibrator_fit=lambda dom, calibrator, non_holdout: fit_seen.__setitem__(
            dom, calibrator
        ),
    )

    # Only the converged domain gets a bootstrap/calibrator-fit callback -- textworld's failed
    # fit must not reach either hook.
    assert set(boot_seen.keys()) == {"tower_of_hanoi"}
    assert set(fit_seen.keys()) == {"tower_of_hanoi"}
    assert "reps" in boot_seen["tower_of_hanoi"]
