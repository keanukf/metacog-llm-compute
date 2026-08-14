"""Canonical real Phase 1 dataset selection.

The real Phase 1 data lives in two directories because of a mid-collection bug and a targeted
re-run, not because the two domains were ever meant to live separately:

- ``data/results/phase1/phase1_20260722_091125/`` -- ``tower_of_hanoi`` (valid) +
  ``textworld`` (**invalid**: 45/50 frozen TextWorld game files were missing at runtime on the
  collection pod, so ``TextWorldEnv`` silently fell back to an unwinnable stub per instance
  instead of erroring; see commit ``b47e35d``, "fix(execution): fail loudly when TextWorld game
  files are missing before a real run"). Confirmed empirically: 5.3% textworld success in this
  directory vs. 53.7% in the valid re-collection below -- consistent with 45/50 instances being
  structurally incapable of success.
- ``data/results/phase1/textworld_regen_20260724/`` -- ``textworld`` only, the valid
  re-collection run after the ``b47e35d`` fix landed.

The canonical real Phase 1 dataset is therefore: ``tower_of_hanoi`` from the first directory +
``textworld`` from the second, 1500 episodes total (750 + 750). This module is the first place
that selection rule is captured in code -- see ``docs/consistency_log.md`` for the dated
evidence-trail entry documenting this (written when this module was added, per project convention
of logging findings at the point they're operationalized, not deferred to a later report).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.analysis.datasets import load_run_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# domain -> the one source directory that domain's episodes are kept from.
CANONICAL_SOURCES: dict[str, Path] = {
    "tower_of_hanoi": REPO_ROOT / "data/results/phase1/phase1_20260722_091125",
    "textworld": REPO_ROOT / "data/results/phase1/textworld_regen_20260724",
}

# Frozen preregistered design (blueprints/gate_p1_readiness.md, both difficulty manifests):
# 2 domains x 50 instances x 5 runs x 3 compute stages.
EXPECTED_STAGES = ("C0", "C1", "C2")
EXPECTED_TOTAL_EPISODES = 1500
EXPECTED_PER_DOMAIN = 750
EXPECTED_PER_DOMAIN_STAGE = 250  # 50 instances x 5 runs
EXPECTED_HOLDOUT_INSTANCES_PER_DOMAIN = 5  # Gate D mod-10 holdout split


@dataclass(frozen=True)
class CanonicalDataset:
    episodes: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    sources: dict[str, str]  # domain -> source run_dir (str, for JSON-friendliness)


def build_canonical_dataset(
    sources: dict[str, Path] | None = None,
) -> CanonicalDataset:
    """Load each source directory once and keep only that directory's designated domain.

    Stamps ``_source_dir`` onto every kept episode so ``assert_canonical_invariants`` can verify
    domain<->source-directory correspondence independently of this function's own filtering logic
    (defense in depth against a future edit to ``CANONICAL_SOURCES`` or this loop silently
    breaking the guarantee).
    """
    sources = sources or CANONICAL_SOURCES
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    source_map: dict[str, str] = {}
    for domain, run_dir in sources.items():
        run_dir = Path(run_dir)
        ds = load_run_dataset(run_dir)
        kept_eps = [e for e in ds.episodes if str(e.get("domain")) == domain]
        for e in kept_eps:
            e["_source_dir"] = str(run_dir)
        kept_ep_ids = {e.get("episode_id") for e in kept_eps}
        kept_steps = [s for s in ds.steps if s.get("episode_id") in kept_ep_ids]
        episodes.extend(kept_eps)
        steps.extend(kept_steps)
        source_map[domain] = str(run_dir)
    return CanonicalDataset(episodes=episodes, steps=steps, sources=source_map)


def assert_canonical_invariants(ds: CanonicalDataset) -> None:
    """Fail loudly (raise) if the canonical dataset deviates from the frozen preregistered design.

    Deliberately raises rather than returning a pass/fail flag -- Stage 0 must stop the pipeline
    dead rather than let a wrong episode count or domain mix-up propagate into every downstream
    confirmatory number, exactly the class of bug that produced the corrupted textworld half of
    ``phase1_20260722_091125`` undetected for two days.
    """
    if len(ds.episodes) != EXPECTED_TOTAL_EPISODES:
        raise AssertionError(
            f"canonical dataset has {len(ds.episodes)} episodes, expected {EXPECTED_TOTAL_EPISODES}"
        )

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in ds.episodes:
        by_domain[str(e.get("domain"))].append(e)

    if set(by_domain.keys()) != set(ds.sources.keys()):
        raise AssertionError(
            f"domains present {sorted(by_domain.keys())} != configured sources "
            f"{sorted(ds.sources.keys())}"
        )

    for domain, expected_dir in ds.sources.items():
        eps = by_domain.get(domain, [])
        if len(eps) != EXPECTED_PER_DOMAIN:
            raise AssertionError(f"{domain}: {len(eps)} episodes, expected {EXPECTED_PER_DOMAIN}")

        wrong_source = [e for e in eps if e.get("_source_dir") != expected_dir]
        if wrong_source:
            raise AssertionError(
                f"{domain}: {len(wrong_source)} episode(s) trace to a source directory other "
                f"than the configured {expected_dir!r} (e.g. {wrong_source[0].get('episode_id')})"
            )

        by_stage: dict[str, int] = defaultdict(int)
        for e in eps:
            by_stage[str(e.get("compute_stage"))] += 1
        if set(by_stage.keys()) != set(EXPECTED_STAGES):
            raise AssertionError(
                f"{domain}: stage set {sorted(by_stage.keys())} != expected {EXPECTED_STAGES}"
            )
        for stage, count in by_stage.items():
            if count != EXPECTED_PER_DOMAIN_STAGE:
                raise AssertionError(
                    f"{domain}/{stage}: {count} episodes, expected {EXPECTED_PER_DOMAIN_STAGE}"
                )

        holdout_instances = {e.get("instance") for e in eps if bool(e.get("holdout"))}
        if len(holdout_instances) != EXPECTED_HOLDOUT_INSTANCES_PER_DOMAIN:
            raise AssertionError(
                f"{domain}: {len(holdout_instances)} holdout instances "
                f"({sorted(holdout_instances, key=str)}), "
                f"expected {EXPECTED_HOLDOUT_INSTANCES_PER_DOMAIN}"
            )


def load_canonical_dataset_from_manifest(manifest_path: str | Path) -> CanonicalDataset:
    """Re-load episode/step rows for exactly the episode IDs recorded in a Stage 0 manifest.

    Every Stage 1+ pipeline script should call this rather than re-deriving the domain/directory
    selection itself (or re-calling ``build_canonical_dataset`` fresh, which would silently
    tolerate the underlying data changing between Stage 0 and a later stage's run) -- reading
    through the manifest's own ``episode_id`` list is what makes "the same Stage 0 output feeds
    every later stage" an actual guarantee rather than a convention.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids_by_source: dict[str, set[str]] = defaultdict(set)
    for entry in manifest["entries"]:
        ids_by_source[entry["source_dir"]].add(entry["episode_id"])

    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    source_map: dict[str, str] = {}
    for source_dir, ep_ids in ids_by_source.items():
        ds = load_run_dataset(source_dir)
        kept = [e for e in ds.episodes if e.get("episode_id") in ep_ids]
        kept_ids = {e.get("episode_id") for e in kept}
        episodes.extend(kept)
        steps.extend(s for s in ds.steps if s.get("episode_id") in kept_ids)
        for e in kept:
            source_map.setdefault(str(e.get("domain")), source_dir)

    return CanonicalDataset(episodes=episodes, steps=steps, sources=source_map)


# --- TextWorld holdout-label correction (docs/consistency_log.md, 2026-08-14 entry) -----------
#
# The canonical Phase 1 TextWorld source (``textworld_regen_20260724``) stamped every episode's
# ``holdout`` field from an uncommitted, non-preregistered instance split -- confirmed empirically
# to be {0,1,2,3,4} (the first 5 instance IDs) rather than the frozen manifest's mod-10 policy
# ({0,10,20,30,40}, ``data/tasks/textworld/difficulty_manifest.json``, git-stable since Gate D).
# Tower of Hanoi is unaffected (its manifest and its episodes' embedded ``holdout`` field agree).
# Every consumer of a ``holdout`` field on TextWorld rows -- Phase 1 (steps) or Phase 2 (episodes)
# -- must apply exactly one of the two corrections below before using it; nothing may trust the
# embedded field for this domain unmodified.
TEXTWORLD_TRUE_HOLDOUT_INSTANCES: frozenset[int] = frozenset({0, 10, 20, 30, 40})

# Phase 2's threshold artifact was fit on the wrong regen-embedded holdout ({0,1,2,3,4}), so
# instances 1-4 were used to fit the deployed adaptive-allocator thresholds *and*, under the
# correct manifest split, are not part of the true holdout -- they legitimately entered Phase 2's
# confirmatory evaluation sample too. That overlap is a real fit/eval circularity for exactly
# those 4 instances (0 is unaffected: both splits agree it is holdout). Any confirmatory or
# exploratory statistic that evaluates the *deployed* Phase 2 policy (H2, the C0/C1 spectrum
# reference) must exclude this larger union, not just the true 5, to remove the overlap.
TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES: frozenset[int] = TEXTWORLD_TRUE_HOLDOUT_INSTANCES | {
    1,
    2,
    3,
    4,
}


def apply_textworld_holdout_correction(
    rows: list[dict[str, Any]], instance_set: frozenset[int]
) -> list[dict[str, Any]]:
    """Overwrite ``holdout`` in place on every TextWorld row (step or episode) to
    ``instance in instance_set``; rows for other domains pass through untouched. Mutates and
    returns ``rows`` for chaining. Use ``TEXTWORLD_TRUE_HOLDOUT_INSTANCES`` when fitting something
    against Phase 1 data (threshold artifact, H1b calibrator) and
    ``TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES`` when evaluating the already-deployed Phase 2
    policy (H2, the C0/C1 spectrum reference) -- see the module-level note above for why these
    differ.
    """
    for r in rows:
        if str(r.get("domain")) == "textworld":
            r["holdout"] = int(r.get("instance")) in instance_set
    return rows
