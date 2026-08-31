"""Phase 1/2 worklist enumeration and resume semantics.

Verifies the job list expands correctly over instances x compute-stages and, on resume, skips
episodes already completed or quarantined. Correct enumeration is what makes a run idempotent:
a re-launched run must reproduce exactly the outstanding set, never re-run or double-count an
episode already contributing to the DV.
"""

from __future__ import annotations

from src.execution.worklist import (
    EpisodeJob,
    build_phase1_worklist,
    build_phase2_worklist,
    group_jobs_by_strategy,
)


def test_phase1_worklist_skips_completed_and_quarantined():
    config = {
        "phase1": {
            "domains": ["textworld"],
            "instances_per_domain": 1,
            "runs_per_condition": 1,
        }
    }
    completed = {"ep_textworld_0_C0_0"}
    quarantined = {"ep_textworld_0_C1_0"}
    jobs = build_phase1_worklist(config, completed=completed, quarantined=quarantined)
    ids = {j.episode_id for j in jobs}
    assert "ep_textworld_0_C0_0" not in ids
    assert "ep_textworld_0_C1_0" not in ids
    assert "ep_textworld_0_C2_0" in ids
    assert len(jobs) == 1


def test_phase1_worklist_enumeration_count():
    config = {
        "phase1": {
            "domains": ["textworld", "tower_of_hanoi"],
            "instances_per_domain": 2,
            "runs_per_condition": 2,
        }
    }
    jobs = build_phase1_worklist(config, completed=set(), quarantined=set())
    assert len(jobs) == 2 * 2 * 3 * 2
    assert all(isinstance(j, EpisodeJob) and j.phase == "phase1" for j in jobs)


def test_phase1_worklist_respects_compute_stages_count():
    config = {
        "phase1": {
            "domains": ["tower_of_hanoi"],
            "instances_per_domain": 20,
            "compute_stages": 1,
            "runs_per_condition": 1,
        }
    }
    jobs = build_phase1_worklist(config, completed=set(), quarantined=set())
    assert len(jobs) == 20
    assert all(j.compute_stage == "C0" for j in jobs)


def test_phase2_worklist_enumeration():
    config = {
        "phase2": {
            "domains": ["textworld"],
            "instances_per_domain": 1,
            "strategies": ["always_c0", "random"],
            "runs_per_condition": 1,
        }
    }
    jobs = build_phase2_worklist(config, completed=set(), quarantined=set())
    assert len(jobs) == 2
    assert jobs[0].strategy in {"always_c0", "random"}


def test_phase2_worklist_is_strategy_major():
    """Strategy must be the outermost grouping so same-strategy jobs are contiguous -- this is
    what lets group_jobs_by_strategy() partition the list into stage-homogeneous blocks without
    interleaving domains/instances of different strategies (docs/consistency_log.md 2026-08-04)."""
    config = {
        "phase2": {
            "domains": ["textworld", "tower_of_hanoi"],
            "instances_per_domain": 2,
            "strategies": ["always_c0", "always_c2", "random"],
            "runs_per_condition": 2,
        }
    }
    jobs = build_phase2_worklist(config, completed=set(), quarantined=set())
    assert len(jobs) == 2 * 2 * 3 * 2
    seen_strategies: list[str] = []
    for job in jobs:
        if not seen_strategies or seen_strategies[-1] != job.strategy:
            seen_strategies.append(str(job.strategy))
    # Each strategy appears as exactly one contiguous run -- no strategy revisited later.
    assert seen_strategies == ["always_c0", "always_c2", "random"]


def test_group_jobs_by_strategy_partitions_contiguous_blocks():
    config = {
        "phase2": {
            "domains": ["textworld"],
            "instances_per_domain": 1,
            "strategies": ["always_c0", "always_c2", "random"],
            "runs_per_condition": 3,
        }
    }
    jobs = build_phase2_worklist(config, completed=set(), quarantined=set())
    groups = group_jobs_by_strategy(jobs)
    assert [g[0].strategy for g in groups] == ["always_c0", "always_c2", "random"]
    assert all(len(g) == 3 for g in groups)
    for g in groups:
        assert all(job.strategy == g[0].strategy for job in g)


def test_group_jobs_by_strategy_empty_list():
    assert group_jobs_by_strategy([]) == []


def test_group_jobs_by_strategy_single_strategy_stays_one_block():
    config = {
        "phase2": {
            "domains": ["textworld"],
            "instances_per_domain": 2,
            "strategies": ["always_c0"],
            "runs_per_condition": 2,
        }
    }
    jobs = build_phase2_worklist(config, completed=set(), quarantined=set())
    groups = group_jobs_by_strategy(jobs)
    assert len(groups) == 1
    assert len(groups[0]) == 4
