"""Phase 2 analysis: Always-C0/C1 exploratory reference
(scripts/phase2_analysis/stage2_c0_c1_reference.py).

CLI smoke test against tiny synthetic Phase 1 manifest + Phase 2 checkpoint fixtures (not the real
data). Verifies: Phase 1 C0/C1 episodes and Phase 2 adaptive/C2 episodes pool correctly into a
five-arm spectrum and pairwise contrasts, and the script fails loudly on missing inputs.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from scripts.phase2_analysis import stage2_c0_c1_reference as stage2
from src.analysis.phase1_canonical import TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES

DOMAINS = ("textworld", "tower_of_hanoi")
# Clear of TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES ({0,1,2,3,4,10,20,30,40}) so these instances
# survive the correction and actually reach the pooled comparison.
SAFE_INSTANCES = [i for i in range(50, 56) if i not in TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES]


def _write_phase1_episode(dir_: Path, *, domain: str, instance: int, stage: str, seed: int) -> None:
    rng = random.Random(seed)
    ep_id = f"ep_{domain}_{instance}_{stage}_0"
    (dir_ / f"{ep_id}.json").write_text(
        json.dumps(
            {
                "episode_id": ep_id,
                "domain": domain,
                "instance": instance,
                "compute_stage": stage,
                "run": 0,
                "holdout": False,
                "task_success": rng.random() < 0.5,
                "total_tokens_generated": rng.randint(100, 5000),
                "episode_length_steps": 4,
                "steps_detail": [],
            }
        ),
        encoding="utf-8",
    )


def _build_phase1_manifest(tmp_path: Path) -> Path:
    source_dir = tmp_path / "phase1_source"
    source_dir.mkdir()
    entries = []
    for domain in DOMAINS:
        for inst in SAFE_INSTANCES:
            for stage in ("C0", "C1"):
                ep_id = f"ep_{domain}_{inst}_{stage}_0"
                _write_phase1_episode(
                    source_dir,
                    domain=domain,
                    instance=inst,
                    stage=stage,
                    seed=inst * 3 + hash(stage) % 5,
                )
                entries.append(
                    {"episode_id": ep_id, "domain": domain, "source_dir": str(source_dir)}
                )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "selection_rule": "test fixture",
                "sources": {d: str(source_dir) for d in DOMAINS},
                "n_episodes": len(entries),
                "content_hash": "test",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_phase2_episode(
    dir_: Path, *, domain: str, instance: int, strategy: str, run: int, seed: int
) -> None:
    rng = random.Random(seed)
    ep_id = f"ep_{domain}_{instance}_{strategy}_{run}"
    (dir_ / f"{ep_id}.json").write_text(
        json.dumps(
            {
                "episode_id": ep_id,
                "domain": domain,
                "instance": instance,
                "strategy": strategy,
                "run": run,
                "holdout": False,
                "task_success": rng.random() < 0.5,
                "total_tokens_generated": rng.randint(100, 5000),
                "episode_length_steps": 4,
                "steps_detail": [],
            }
        ),
        encoding="utf-8",
    )


def _build_phase2_checkpoint(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    seed = 0
    for domain in DOMAINS:
        for inst in SAFE_INSTANCES:
            for strategy in ("adaptive_tle", "adaptive_vc", "always_c2"):
                for run in range(2):
                    seed += 1
                    _write_phase2_episode(
                        ckpt_dir,
                        domain=domain,
                        instance=inst,
                        strategy=strategy,
                        run=run,
                        seed=seed,
                    )
    return ckpt_dir


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["stage2_c0_c1_reference.py"] + argv)
    return stage2.main()


def test_writes_five_arm_spectrum_and_pairwise_contrasts(tmp_path, monkeypatch, capsys):
    manifest_path = _build_phase1_manifest(tmp_path)
    checkpoint_dir = _build_phase2_checkpoint(tmp_path)
    out_path = tmp_path / "c0_c1.json"
    figures_dir = tmp_path / "figures"

    rc = _run_main(
        [
            "--phase1-manifest",
            str(manifest_path),
            "--phase2-checkpoint-dir",
            str(checkpoint_dir),
            "--output",
            str(out_path),
            "--figures-output",
            str(figures_dir),
            "--n-boot",
            "50",
        ],
        monkeypatch,
    )
    assert rc == 0
    assert out_path.exists()

    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert obj["status"] == "exploratory_only"
    for domain in DOMAINS:
        spectrum = obj["by_domain"][domain]["spectrum"]
        for arm in ("always_c0", "always_c1", "adaptive_tle", "adaptive_vc", "always_c2"):
            assert spectrum[arm]["n_instances"] == len(SAFE_INSTANCES)
        pairwise = obj["by_domain"][domain]["pairwise_vs_reference"]
        assert pairwise["adaptive_tle_vs_always_c0"]["n_pairs"] == len(SAFE_INSTANCES)

    captured = capsys.readouterr()
    assert "Stage 2 OK -- C0/C1 reference written to" in captured.out


def test_fails_loudly_on_missing_manifest(tmp_path, monkeypatch):
    rc = _run_main(
        [
            "--phase1-manifest",
            str(tmp_path / "missing.json"),
            "--phase2-checkpoint-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.json"),
        ],
        monkeypatch,
    )
    assert rc == 1


def test_fails_loudly_on_missing_checkpoint_dir(tmp_path, monkeypatch):
    manifest_path = _build_phase1_manifest(tmp_path)
    rc = _run_main(
        [
            "--phase1-manifest",
            str(manifest_path),
            "--phase2-checkpoint-dir",
            str(tmp_path / "missing_checkpoint"),
            "--output",
            str(tmp_path / "out.json"),
        ],
        monkeypatch,
    )
    assert rc == 1
