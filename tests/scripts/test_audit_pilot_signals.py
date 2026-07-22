"""Pilot signal-availability audit (``audit_pilot_signals.py``).

Verifies the audit reports metacognitive-signal presence (TLE, VC) broken out by compute stage
across pilot episodes. This is a pre-analysis gate: RQ1 calibration needs both signals populated
at the expected stages, so the audit surfaces missing-signal gaps before they silently shrink the
usable sample.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.pilot_analysis.audit_pilot_signals import audit


def test_audit_reports_signals_by_stage(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot_test"
    pilot_dir.mkdir()
    ep = {
        "domain": "textworld",
        "success": True,
        "stage_per_step": ["C0", "C1", "C2"],
        "vc_per_step": [50.0, 60.0, None],
        "tle_per_step": [{"mean_entropy": 0.1}, None, {"mean_entropy": 0.2}],
    }
    (pilot_dir / "ep_textworld_0.json").write_text(json.dumps(ep), encoding="utf-8")
    (pilot_dir / "pilot_sanity.json").write_text(
        json.dumps(
            {
                "has_logprobs": True,
                "completion_tokens_observed": 5,
                "logprob_token_field_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    trace = {
        "compute_stage": "C2",
        "call_detail": {
            "method": "self_consistency_majority_vote",
            "subcalls": [
                {"kind": "sample", "sample_index": 0},
                {"kind": "sample", "sample_index": 1},
                {"kind": "sample", "sample_index": 2},
            ],
        },
        "response_full": "go north",
    }
    (pilot_dir / "trace_ep_0.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")

    report = audit(pilot_dir)
    by_stage = report["signals_by_stage"]
    assert by_stage["C0"]["tle_rate"] == 1.0
    assert by_stage["C1"]["tle_rate"] == 0.0
    c2 = report["c2_trace_audit"]
    assert c2["c2_steps_seen"] == 1
    assert c2["c2_steps_with_majority_vote_method"] == 1
