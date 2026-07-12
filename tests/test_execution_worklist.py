"""Tests for execution worklist enumeration."""

from __future__ import annotations

from src.execution.worklist import EpisodeJob, build_phase1_worklist, build_phase2_worklist


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
