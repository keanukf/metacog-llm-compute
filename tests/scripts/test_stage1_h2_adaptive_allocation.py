"""Phase 2 analysis: H2 adaptive allocation
(scripts/phase2_analysis/stage1_h2_adaptive_allocation.py).

CLI smoke test against a tiny synthetic Phase 2 checkpoint fixture (not the real ~175 MB Phase 2
dataset). Verifies: the artifact is written with real paired H2 statistics for both domains, the
TextWorld holdout correction actually removes the four contaminated instances from the
confirmatory sample, and the script fails loudly on a missing checkpoint dir.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from scripts.phase2_analysis import stage1_h2_adaptive_allocation as stage1
from src.analysis.phase1_canonical import TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES

STRATEGIES = ("adaptive_tle", "adaptive_vc", "always_c2")
# Clear of TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES ({0,1,2,3,4,10,20,30,40}) so these instances
# survive the correction and actually reach the confirmatory sample.
TEXTWORLD_SAFE_INSTANCES = [
    i for i in range(50, 56) if i not in TEXTWORLD_CONFIRMATORY_EXCLUDED_INSTANCES
]
TOWER_OF_HANOI_INSTANCES = list(range(6))  # domain is unaffected by the correction


def _write_episode(
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
                "task_success": rng.random() < (0.7 if strategy == "always_c2" else 0.4),
                "total_tokens_generated": rng.randint(100, 5000),
                "episode_length_steps": 4,
                "steps_detail": [],
            }
        ),
        encoding="utf-8",
    )


def _build_checkpoint(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    seed = 0
    for domain, instances in (
        ("textworld", TEXTWORLD_SAFE_INSTANCES),
        ("tower_of_hanoi", TOWER_OF_HANOI_INSTANCES),
    ):
        for inst in instances:
            for strategy in STRATEGIES:
                for run in range(3):
                    seed += 1
                    _write_episode(
                        ckpt_dir,
                        domain=domain,
                        instance=inst,
                        strategy=strategy,
                        run=run,
                        seed=seed,
                    )
    return ckpt_dir


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["stage1_h2_adaptive_allocation.py"] + argv)
    return stage1.main()


def test_writes_paired_h2_statistics_for_both_domains(tmp_path, monkeypatch, capsys):
    checkpoint_dir = _build_checkpoint(tmp_path)
    out_path = tmp_path / "h2.json"
    figures_dir = tmp_path / "figures"

    rc = _run_main(
        [
            "--checkpoint-dir",
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
    assert obj["confirmatory_domain"] == "textworld"
    assert obj["exploratory_domain"] == "tower_of_hanoi"

    tw = obj["by_domain"]["textworld"]
    assert tw["adaptive_tle"]["n_pairs"] == len(TEXTWORLD_SAFE_INSTANCES)
    assert tw["adaptive_tle"]["holm"] is not None

    toh = obj["by_domain"]["tower_of_hanoi"]
    assert toh["adaptive_tle"]["n_pairs"] == len(TOWER_OF_HANOI_INSTANCES)
    assert toh["adaptive_tle"]["holm"] is None
    assert toh["adaptive_tle"]["status"] == "exploratory_reduced_n"

    captured = capsys.readouterr()
    assert "Stage 1 OK -- H2 written to" in captured.out
    assert "CONFIRMATORY" in captured.out
    assert "EXPLORATORY" in captured.out


def test_fails_loudly_on_missing_checkpoint_dir(tmp_path, monkeypatch):
    rc = _run_main(
        [
            "--checkpoint-dir",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "out.json"),
        ],
        monkeypatch,
    )
    assert rc == 1
