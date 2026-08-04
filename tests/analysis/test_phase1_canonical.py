"""Canonical Phase 1 dataset selection (``src/analysis/phase1_canonical.py``).

Verifies the two-directory domain selection actually discards wrong-domain contamination -- the
exact failure mode that would silently re-admit the corrupted textworld half of
``phase1_20260722_091125`` if the per-domain filter ever regressed -- that invariant violations
raise loudly rather than passing silently, and that ``build_canonical_dataset``'s output is
stable across repeated calls against the same input directories (idempotence).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.phase1_canonical import (
    CanonicalDataset,
    assert_canonical_invariants,
    build_canonical_dataset,
    load_canonical_dataset_from_manifest,
)


def _write_episode(
    dir_: Path,
    *,
    episode_id: str,
    domain: str,
    instance: int,
    compute_stage: str,
    holdout: bool,
) -> None:
    (dir_ / f"{episode_id}.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "domain": domain,
                "instance": instance,
                "compute_stage": compute_stage,
                "run": 0,
                "holdout": holdout,
                "task_success": True,
                "episode_length_steps": 0,
                "steps_detail": [],
            }
        )
    )


def _build_mixed_domain_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """dir1 mostly tower_of_hanoi with 2 contaminating textworld episodes (mirrors the real
    corrupted phase1_20260722_091125 dir, which does contain real textworld files -- just
    invalid ones); dir2 mostly textworld with 2 contaminating tower_of_hanoi episodes."""
    dir1 = tmp_path / "dir1_toh_and_bad_textworld"
    dir2 = tmp_path / "dir2_textworld_regen"
    dir1.mkdir()
    dir2.mkdir()
    for i in range(3):
        _write_episode(
            dir1,
            episode_id=f"ep_tower_of_hanoi_{i}_C0_0",
            domain="tower_of_hanoi",
            instance=i,
            compute_stage="C0",
            holdout=(i == 0),
        )
    for i in range(2):
        _write_episode(
            dir1,
            episode_id=f"ep_textworld_{i}_C0_0",
            domain="textworld",
            instance=i,
            compute_stage="C0",
            holdout=False,
        )
    for i in range(3):
        _write_episode(
            dir2,
            episode_id=f"ep_textworld_{i}_C0_0",
            domain="textworld",
            instance=i,
            compute_stage="C0",
            holdout=(i == 0),
        )
    for i in range(2):
        _write_episode(
            dir2,
            episode_id=f"ep_tower_of_hanoi_{i}_C0_0",
            domain="tower_of_hanoi",
            instance=i,
            compute_stage="C0",
            holdout=False,
        )
    return dir1, dir2


def test_build_canonical_dataset_discards_wrong_domain_contamination(tmp_path):
    dir1, dir2 = _build_mixed_domain_dirs(tmp_path)
    ds = build_canonical_dataset({"tower_of_hanoi": dir1, "textworld": dir2})
    domains = {e["domain"] for e in ds.episodes}
    assert domains == {"tower_of_hanoi", "textworld"}
    assert len([e for e in ds.episodes if e["domain"] == "tower_of_hanoi"]) == 3
    assert len([e for e in ds.episodes if e["domain"] == "textworld"]) == 3
    assert len(ds.episodes) == 6  # the 2+2 wrong-domain contaminating episodes must not survive
    for e in ds.episodes:
        expected_dir = str(dir1) if e["domain"] == "tower_of_hanoi" else str(dir2)
        assert e["_source_dir"] == expected_dir


def test_build_canonical_dataset_is_idempotent(tmp_path):
    dir1, dir2 = _build_mixed_domain_dirs(tmp_path)
    sources = {"tower_of_hanoi": dir1, "textworld": dir2}
    ids1 = sorted(e["episode_id"] for e in build_canonical_dataset(sources).episodes)
    ids2 = sorted(e["episode_id"] for e in build_canonical_dataset(sources).episodes)
    assert ids1 == ids2


def test_assert_canonical_invariants_passes_on_correctly_sized_data(tmp_path, monkeypatch):
    import src.analysis.phase1_canonical as pc

    monkeypatch.setattr(pc, "EXPECTED_TOTAL_EPISODES", 6)
    monkeypatch.setattr(pc, "EXPECTED_PER_DOMAIN", 3)
    monkeypatch.setattr(pc, "EXPECTED_PER_DOMAIN_STAGE", 3)
    monkeypatch.setattr(pc, "EXPECTED_STAGES", ("C0",))  # fixture only uses C0 for simplicity
    monkeypatch.setattr(pc, "EXPECTED_HOLDOUT_INSTANCES_PER_DOMAIN", 1)
    dir1, dir2 = _build_mixed_domain_dirs(tmp_path)
    ds = pc.build_canonical_dataset({"tower_of_hanoi": dir1, "textworld": dir2})
    pc.assert_canonical_invariants(ds)  # must not raise


def test_load_canonical_dataset_from_manifest_reproduces_build_output(tmp_path):
    """Every Stage 1+ script reads through a Stage 0 manifest rather than re-deriving the
    domain/directory split -- this must reconstruct exactly the same episode set."""
    dir1, dir2 = _build_mixed_domain_dirs(tmp_path)
    sources = {"tower_of_hanoi": dir1, "textworld": dir2}
    original = build_canonical_dataset(sources)
    manifest = {
        "entries": [
            {"episode_id": e["episode_id"], "domain": e["domain"], "source_dir": e["_source_dir"]}
            for e in original.episodes
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    reloaded = load_canonical_dataset_from_manifest(manifest_path)
    assert sorted(e["episode_id"] for e in reloaded.episodes) == sorted(
        e["episode_id"] for e in original.episodes
    )
    assert reloaded.sources == {"tower_of_hanoi": str(dir1), "textworld": str(dir2)}


def test_assert_canonical_invariants_raises_on_wrong_total_count():
    ds = CanonicalDataset(episodes=[], steps=[], sources={"tower_of_hanoi": "x", "textworld": "y"})
    with pytest.raises(AssertionError, match="episodes"):
        assert_canonical_invariants(ds)


def test_assert_canonical_invariants_raises_on_domain_source_mismatch(monkeypatch):
    import src.analysis.phase1_canonical as pc

    monkeypatch.setattr(pc, "EXPECTED_TOTAL_EPISODES", 1)
    monkeypatch.setattr(pc, "EXPECTED_PER_DOMAIN", 1)
    ds = CanonicalDataset(
        episodes=[
            {
                "domain": "tower_of_hanoi",
                "instance": 0,
                "compute_stage": "C0",
                "holdout": False,
                "_source_dir": "wrong-dir",
            }
        ],
        steps=[],
        sources={"tower_of_hanoi": "expected-dir"},
    )
    with pytest.raises(AssertionError, match="source directory"):
        assert_canonical_invariants(ds)
