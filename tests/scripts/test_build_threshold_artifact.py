"""Phase 2 prep: threshold/policy artifact builder (scripts/phase2_prep/build_threshold_artifact.py).

CLI smoke test against a tiny synthetic manifest + episode fixture (not the real ~24 GB Phase 1
dataset). Verifies: the artifact is written with a real (non-fallback) grid-searched policy when
holdout data is present, fails loudly on a missing manifest, and fails loudly (rather than
silently writing a legacy-fallback artifact) when no holdout steps exist at all.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from scripts.phase2_prep import build_threshold_artifact

DOMAIN = "textworld"


def _write_episode_with_steps(
    dir_: Path, *, episode_id: str, instance: int, holdout: bool, stage: str, seed: int
) -> None:
    rng = random.Random(seed)
    steps_detail = []
    for i in range(6):
        tle = rng.uniform(0.0, 1.0)
        p_correct = max(0.05, min(0.95, 1.0 - tle))
        correct = rng.random() < p_correct
        steps_detail.append(
            {
                "step_index": i,
                "compute_stage": stage,
                "tle": {"mean_entropy": tle},
                "vc": rng.uniform(0.0, 100.0),
                "correctness": "optimal" if correct else "illegal",
            }
        )
    (dir_ / f"{episode_id}.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "domain": DOMAIN,
                "instance": instance,
                "compute_stage": stage,
                "run": 0,
                "holdout": holdout,
                "task_success": True,
                "episode_length_steps": 6,
                "steps_detail": steps_detail,
            }
        ),
        encoding="utf-8",
    )


def _build_manifest(tmp_path: Path, *, n_holdout_instances: int, n_pool_instances: int) -> Path:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    entries = []
    stages = ("C0", "C1", "C2")
    inst = 0
    for holdout, n in ((True, n_holdout_instances), (False, n_pool_instances)):
        for _ in range(n):
            for stage in stages:
                ep_id = f"ep_{DOMAIN}_{inst}_{stage}_0"
                _write_episode_with_steps(
                    source_dir,
                    episode_id=ep_id,
                    instance=inst,
                    holdout=holdout,
                    stage=stage,
                    seed=inst * 7 + hash(stage) % 5,
                )
                entries.append(
                    {"episode_id": ep_id, "domain": DOMAIN, "source_dir": str(source_dir)}
                )
            inst += 1

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "selection_rule": "test fixture",
                "sources": {DOMAIN: str(source_dir)},
                "n_episodes": len(entries),
                "content_hash": "test",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_threshold_artifact.py"] + argv)
    return build_threshold_artifact.main()


def test_writes_real_grid_searched_artifact(tmp_path, monkeypatch, capsys):
    # 5 holdout instances x 3 stages x 6 steps = 90 holdout steps, comfortably clears the
    # total_holdout < 10 "insufficient" guard in grid_search_thresholds.
    manifest_path = _build_manifest(tmp_path, n_holdout_instances=5, n_pool_instances=5)
    out_path = tmp_path / "threshold_artifact.json"

    rc = _run_main(["--manifest", str(manifest_path), "--output", str(out_path)], monkeypatch)
    assert rc == 0
    assert out_path.exists()

    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert DOMAIN in obj["by_domain"]
    for signal in ("vc", "tle_mean_entropy"):
        block = obj["by_domain"][DOMAIN][signal]
        assert block.get("note") != "insufficient holdout samples"
        assert block["theta1"] is not None
        assert block["theta2"] is not None

    captured = capsys.readouterr()
    assert "OK -- threshold artifact written to" in captured.out
    assert f"{DOMAIN}/tle_mean_entropy" in captured.out or f"{DOMAIN}/vc" in captured.out


def test_fails_loudly_on_missing_manifest(tmp_path, monkeypatch):
    rc = _run_main(
        ["--manifest", str(tmp_path / "missing.json"), "--output", str(tmp_path / "out.json")],
        monkeypatch,
    )
    assert rc == 1


def test_fails_loudly_when_no_holdout_steps_exist(tmp_path, monkeypatch):
    # All non-holdout -> the canonical dataset has zero holdout steps.
    manifest_path = _build_manifest(tmp_path, n_holdout_instances=0, n_pool_instances=5)
    out_path = tmp_path / "threshold_artifact.json"

    rc = _run_main(["--manifest", str(manifest_path), "--output", str(out_path)], monkeypatch)
    assert rc == 1
    assert not out_path.exists()
