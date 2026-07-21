"""Episode quarantine classification and the pod autostop wrapper.

Verifies exclusion reasons map to the preregistered codes, that writing an unknown reason is
rejected, the quarantine/resume-skip roundtrip, and that the autostop shell wrapper carries the
RunPod pod-id hook. Restricting quarantine to preregistered codes is a DV-protection rule: only
prespecified infrastructure faults may exclude an episode, so an ad-hoc reason can't be used to
drop an inconvenient-but-valid result post hoc.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.errors import EnvStateError, LabelError
from src.utils.run_resilience import (
    classify_exclusion_reason,
    load_quarantined_episode_ids,
    write_quarantine,
)


def test_classify_exclusion_reason_maps_prereg_codes():
    assert classify_exclusion_reason(EnvStateError("bad step")) == "env_assertion"
    assert classify_exclusion_reason(LabelError("unreachable")) == "label_error"
    assert classify_exclusion_reason(RuntimeError("other")) is None


def test_write_quarantine_rejects_unknown_reason(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported quarantine reason"):
        write_quarantine(tmp_path, "ep_x", "unknown")


def test_quarantine_roundtrip_and_resume_skip_set(tmp_path: Path):
    write_quarantine(
        tmp_path,
        "ep_tw_0_C1_0",
        "env_assertion",
        meta={"domain": "textworld", "instance": 0, "stage_or_strategy": "C1"},
    )
    write_quarantine(
        tmp_path,
        "ep_toh_1_C2_0",
        "label_error",
        meta={"domain": "tower_of_hanoi", "instance": 1, "stage_or_strategy": "C2"},
    )
    assert load_quarantined_episode_ids(tmp_path) == {"ep_tw_0_C1_0", "ep_toh_1_C2_0"}
    lines = (tmp_path / "quarantine.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["reason"] == "env_assertion"
    assert first["domain"] == "textworld"


def test_autostop_wrapper_script_exists_and_has_pod_hook():
    script = (
        Path(__file__).resolve().parent.parent.parent
        / "scripts"
        / "cloud"
        / "shell"
        / "run_with_autostop.sh"
    )
    assert script.is_file()
    text = script.read_text()
    assert "RUNPOD_POD_ID" in text
    assert 'runpodctl stop pod "$POD_ID"' in text
    assert 'python "$@"' in text
