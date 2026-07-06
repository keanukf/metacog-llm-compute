"""Policy artifact write/load roundtrip on real holdout step rows (not test fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.allocation_policy import load_policy
from src.analysis.thresholds import write_threshold_artifact

STEPS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "policy_roundtrip_steps.json"


@pytest.fixture
def holdout_steps() -> list[dict]:
    if not STEPS_FIXTURE.is_file():
        pytest.skip(
            f"Missing real step fixture {STEPS_FIXTURE.name}: flatten steps from Pod smoke "
            "(configs/dev/smoke.yaml), check in as tests/fixtures/policy_roundtrip_steps.json"
        )
    raw = json.loads(STEPS_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and raw, "fixture must be a non-empty JSON list of step rows"
    return raw


def test_policy_artifact_roundtrip_from_real_steps(
    tmp_path: Path, holdout_steps: list[dict]
) -> None:
    assert any(bool(r.get("holdout")) for r in holdout_steps), (
        "fixture must include holdout rows so grid-search path is used"
    )

    artifact_path = write_threshold_artifact(tmp_path / "policy.json", holdout_steps)
    body = json.loads(artifact_path.read_text(encoding="utf-8"))

    domain = str(holdout_steps[0].get("domain", "textworld"))
    signal_block = body["by_domain"][domain]["tle_mean_entropy"]
    assert signal_block["objective_definition"] == "step_level_proxy_v1"

    pol = load_policy(artifact_path, domain=domain, signal="tle_mean_entropy")
    ref = pol.ecdf_ref
    assert len(ref) >= 3, "ECDF reference needs at least three values for low/mid/high stage checks"

    low, mid, high = ref[0], ref[len(ref) // 2], ref[-1]
    stages = {pol.stage(low), pol.stage(mid), pol.stage(high)}
    assert stages.issubset({"C0", "C1", "C2"})
