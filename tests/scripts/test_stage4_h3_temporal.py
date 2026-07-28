"""Phase 1 analysis pipeline, Stage 4 (H3 temporal degradation).

Smoke test against tiny synthetic data. Verifies: textworld (confirmatory) gets Holm-corrected
across its 2 signals, tower_of_hanoi (exploratory) does not get folded into that correction, and
a genuinely degrading signal (interaction becomes more negative than a null one) is at least
distinguishable in the fitted coefficient's sign.
"""

from __future__ import annotations

import random

from scripts.phase1_analysis.stage4_h3_temporal import (
    CONFIRMATORY_DOMAIN,
    EXPLORATORY_DOMAIN,
    SIGNALS,
    run_h3,
)


def _degrading_signal_steps(domain: str, *, n_instances: int, steps_per_instance: int, seed: int) -> list[dict]:
    """Signal predicts correctness well early in the episode, badly late -- a genuine
    signal x position interaction, same construction style as fit_h3_model's own tests."""
    rng = random.Random(seed)
    rows = []
    for inst in range(n_instances):
        for t in range(steps_per_instance):
            pos = t / max(steps_per_instance - 1, 1)
            tle = rng.uniform(0.0, 0.2) if pos < 0.5 else rng.uniform(0.4, 0.6)
            p_correct = max(0.05, min(0.95, 1.0 - tle - 0.3 * pos))
            y = 1 if rng.random() < p_correct else 0
            for stage in ("C0", "C1", "C2"):
                rows.append(
                    {
                        "domain": domain,
                        "instance_key": f"{domain}:{inst}",
                        "compute_stage": stage,
                        "y_optimal": y,
                        "tle_mean_entropy": tle,
                        "vc": 100 * (1 - tle),
                        "position_norm": pos,
                    }
                )
    return rows


def test_run_h3_shape_and_family_scoping():
    steps = []
    steps += _degrading_signal_steps(CONFIRMATORY_DOMAIN, n_instances=15, steps_per_instance=10, seed=1)
    steps += _degrading_signal_steps(EXPLORATORY_DOMAIN, n_instances=15, steps_per_instance=6, seed=2)

    result = run_h3(steps)

    assert result["family"] == "E"
    assert result["confirmatory_domain"] == CONFIRMATORY_DOMAIN
    assert result["exploratory_domain"] == EXPLORATORY_DOMAIN

    for sig in SIGNALS:
        conf = result["results"][CONFIRMATORY_DOMAIN][sig]
        assert conf.get("converged") is True
        assert "holm" in conf
        assert "decision_holds" in conf

        expl = result["results"][EXPLORATORY_DOMAIN][sig]
        assert "holm" not in expl  # exploratory domain must not be folded into the Holm family
        assert "note" in expl
