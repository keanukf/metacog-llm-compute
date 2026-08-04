"""Phase 1 analysis pipeline, run_all.py orchestrator.

Doesn't execute the real (multi-minute) stage scripts -- monkeypatches subprocess.run to verify
the control-flow contract: stages run in order, and a non-zero exit stops the chain immediately
without running later stages.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from scripts.phase1_analysis import run_all


def _run_main(argv, monkeypatch, returncodes):
    calls = []

    def fake_run(cmd, cwd=None):
        calls.append(cmd)
        idx = len(calls) - 1
        rc = returncodes[idx] if idx < len(returncodes) else 0
        return SimpleNamespace(returncode=rc)

    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_all.py"] + argv)
    rc = run_all.main()
    return rc, calls


def test_run_all_runs_all_stages_in_order_on_success(monkeypatch):
    rc, calls = _run_main([], monkeypatch, returncodes=[0] * 8)
    assert rc == 0
    assert len(calls) == 8
    scripts = [str(c[1]) for c in calls]
    assert scripts[0].endswith("stage0_build_canonical_dataset.py")
    assert scripts[-1].endswith("stage7_generate_report.py")


def test_run_all_stops_at_first_failure(monkeypatch):
    rc, calls = _run_main([], monkeypatch, returncodes=[0, 0, 1])
    assert rc == 1
    assert len(calls) == 3  # stages 0, 1, 2 attempted; 3-7 never run


def test_run_all_threads_seed_and_n_boot_into_bootstrap_stages(monkeypatch):
    _rc, calls = _run_main(["--seed", "42", "--n-boot", "100"], monkeypatch, returncodes=[0] * 8)
    stage2_cmd = calls[2]
    assert "--seed" in stage2_cmd and "42" in stage2_cmd
    assert "--n-boot" in stage2_cmd and "100" in stage2_cmd
