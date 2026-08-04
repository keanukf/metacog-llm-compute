"""Phase 1 analysis pipeline, Stage 2 (H1a discrimination).

Smoke test against tiny synthetic data (not the real ~45k-row dataset, which takes minutes per
cluster-bootstrap call) -- verifies the JSON output shape, that the confirmatory decision and
Holm correction are wired correctly, and that the descriptive cross-check is computed per domain
(not pooled across domains, which would blend a strong signal in one domain with a null one in
the other and make the comparison meaningless).
"""

from __future__ import annotations

from scripts.phase1_analysis.stage2_h1a_discrimination import DOMAINS, run_h1a


def _make_step(domain: str, instance: int, i: int, *, tle_discriminates: bool) -> dict:
    y = i % 2
    # TLE: lower entropy -> higher score (delta_auroc negates tle_mean_entropy internally) --
    # make it perfectly discriminating when tle_discriminates, uninformative otherwise.
    tle = (0.1 if y == 1 else 0.9) if tle_discriminates else 0.5
    return {
        "domain": domain,
        "instance_key": f"{domain}:{instance}",
        "y_optimal": y,
        "tle_mean_entropy": tle,
        "vc": 50,  # constant -> uninformative VC in both domains
    }


def _make_episode(domain: str, instance: int, *, tle_discriminates: bool) -> dict:
    steps_detail = []
    for i in range(4):
        correct = i % 2 == 0
        tle_val = (0.1 if correct else 0.9) if tle_discriminates else 0.5
        steps_detail.append(
            {
                "correctness": "optimal" if correct else "illegal",
                "tle": {"mean_entropy": tle_val},
                "vc": 50,
            }
        )
    return {"domain": domain, "instance": instance, "steps_detail": steps_detail}


def test_run_h1a_shape_and_per_domain_descriptive_split():
    steps = []
    episodes = []
    for inst in range(6):
        # tower_of_hanoi: TLE genuinely discriminates; textworld: it doesn't.
        for i in range(10):
            steps.append(_make_step("tower_of_hanoi", inst, i, tle_discriminates=True))
            steps.append(_make_step("textworld", inst, i, tle_discriminates=False))
        episodes.append(_make_episode("tower_of_hanoi", inst, tle_discriminates=True))
        episodes.append(_make_episode("textworld", inst, tle_discriminates=False))

    result = run_h1a(steps, episodes, n_boot=200, seed=1)

    assert result["family"] == "A"
    assert set(result["by_domain"].keys()) == set(DOMAINS)
    for dom in DOMAINS:
        d = result["by_domain"][dom]
        assert "point" in d and "ci_low" in d and "ci_high" in d
        assert "decision_holds" in d
        assert "one_sided_pvalue" in d
        assert "holm" in d
        assert "reps" not in d  # stripped before persisting -- raw replicates aren't a report field

    # tower_of_hanoi's TLE genuinely discriminates -> should hold; textworld's doesn't -> shouldn't.
    assert result["by_domain"]["tower_of_hanoi"]["decision_holds"] is True
    assert result["by_domain"]["textworld"]["decision_holds"] is False

    # descriptive cross-check must be split per domain, not one pooled report
    assert set(result["descriptive_cross_check"].keys()) == set(DOMAINS)
    toh_desc = result["descriptive_cross_check"]["tower_of_hanoi"]["optimal_only"]["tle"]["auroc"]
    tw_desc = result["descriptive_cross_check"]["textworld"]["optimal_only"]["tle"]["auroc"]
    assert toh_desc > tw_desc  # matches the confirmatory finding's direction


def test_run_h1a_on_bootstrap_hook_fires_once_per_domain_with_reps():
    steps = []
    for inst in range(6):
        for i in range(10):
            steps.append(_make_step("tower_of_hanoi", inst, i, tle_discriminates=True))
            steps.append(_make_step("textworld", inst, i, tle_discriminates=False))

    seen: dict[str, dict] = {}

    def _on_bootstrap(dom, boot):
        seen[dom] = boot

    run_h1a(steps, [], n_boot=50, seed=1, on_bootstrap=_on_bootstrap)

    assert set(seen.keys()) == set(DOMAINS)
    for boot in seen.values():
        assert "reps" in boot and len(boot["reps"]) > 0
