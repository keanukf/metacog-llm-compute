"""Episode job enumeration for Phase 1 / Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PhaseKind = Literal["phase1", "phase2"]


@dataclass(frozen=True)
class EpisodeJob:
    episode_id: str
    domain: str
    instance: int
    run: int
    phase: PhaseKind
    compute_stage: str | None = None
    strategy: str | None = None


def build_phase1_worklist(
    config: dict[str, Any],
    *,
    completed: set[str],
    quarantined: set[str],
) -> list[EpisodeJob]:
    phase1 = config.get("phase1", {})
    domains = list(phase1.get("domains", ["textworld", "tower_of_hanoi"]))
    instances_per_domain = int(phase1.get("instances_per_domain", 50))
    stages = ["C0", "C1", "C2"]
    runs = int(phase1.get("runs_per_condition", 5))
    jobs: list[EpisodeJob] = []
    for domain in domains:
        for inst in range(instances_per_domain):
            for stage in stages:
                for run in range(runs):
                    ep_id = f"ep_{domain}_{inst}_{stage}_{run}"
                    if ep_id in completed or ep_id in quarantined:
                        continue
                    jobs.append(
                        EpisodeJob(
                            episode_id=ep_id,
                            domain=str(domain),
                            instance=int(inst),
                            run=int(run),
                            phase="phase1",
                            compute_stage=str(stage),
                        )
                    )
    return jobs


def build_phase2_worklist(
    config: dict[str, Any],
    *,
    completed: set[str],
    quarantined: set[str],
) -> list[EpisodeJob]:
    phase2 = config.get("phase2", {})
    domains = list(phase2.get("domains", ["textworld", "tower_of_hanoi"]))
    instances_per_domain = int(phase2.get("instances_per_domain", 50))
    strategies = list(
        phase2.get(
            "strategies",
            ["adaptive_tle", "always_c0", "always_c2", "random", "eager_style", "adaptive_vc"],
        )
    )
    runs = int(phase2.get("runs_per_condition", 5))
    jobs: list[EpisodeJob] = []
    for domain in domains:
        for inst in range(instances_per_domain):
            for strategy in strategies:
                for run in range(runs):
                    ep_id = f"ep_{domain}_{inst}_{strategy}_{run}"
                    if ep_id in completed or ep_id in quarantined:
                        continue
                    jobs.append(
                        EpisodeJob(
                            episode_id=ep_id,
                            domain=str(domain),
                            instance=int(inst),
                            run=int(run),
                            phase="phase2",
                            strategy=str(strategy),
                        )
                    )
    return jobs


def expected_episode_ids(jobs: list[EpisodeJob]) -> list[str]:
    return [j.episode_id for j in jobs]
