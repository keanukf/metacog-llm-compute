"""PR2: random allocation reproducibility, VC judged context, history guard, calibration."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from src.agent.allocator import allocate
from src.agent.stages.shared import (
    _build_model_output_to_judge_section,
    _build_vc_followup_prompt,
    _run_vc_followup,
)
from src.analysis.calibration import compare_signal_calibration
from src.utils.history_guard import enforce_full_history_or_exit, history_truncation_active

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_random_allocate_reproducible_by_episode_seed():
    ep_id = "ep_textworld_3_random_2"
    seed = int(hashlib.md5(ep_id.encode()).hexdigest()[:8], 16)
    import random

    rng1 = random.Random(seed)
    rng2 = random.Random(seed)
    seq1 = [allocate(None, "random", i, rng1) for i in range(8)]
    seq2 = [allocate(None, "random", i, rng2) for i in range(8)]
    assert seq1 == seq2


def test_vc_judged_context_action_only():
    block = _build_model_output_to_judge_section(
        "C1",
        "go north",
        judged_context="action_only",
        raw_action_completion="long cot...",
        cot_text="should not appear",
        verify_completion=None,
        c2_n_samples=None,
        c2_sample_first_lines=None,
        followup_cot_max_chars=100,
        raw_completion_max_chars=100,
    )
    assert block == "[C1] go north"
    assert "cot" not in block.lower()


def test_vc_followup_prompt_contains_action_only_block():
    prompt = _build_vc_followup_prompt(
        "obs",
        [],
        "prefix",
        stage_tag="C2",
        action_line="A->C",
        instruction="Confidence:",
        judged_context="action_only",
    )
    assert "<output_to_judge>" in prompt
    assert "[C2] A->C" in prompt


def test_vc_retry_once_on_parse_failure():
    class _RetryModel:
        def __init__(self) -> None:
            self.calls = 0
            self.temperatures: list[float] = []

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.calls += 1
            self.temperatures.append(float(kwargs.get("temperature", 0.0)))
            if self.calls == 1:
                return "unparseable garbage", None
            return "85", None

    model = _RetryModel()
    vc, detail, extra_tok, extra_calls = _run_vc_followup(
        model,
        observation="obs",
        history=[],
        prompt_prefix="prefix",
        stage_tag="C1",
        action_line="go north",
        vc_followup_instruction="Confidence 0-100:",
        judged_context="action_only",
        retry_on_parse_failure=True,
        followup_max_tokens=16,
        followup_temperature=0.7,
        request_logprobs=False,
    )
    assert model.calls == 2
    assert extra_calls == 2
    assert vc == 85.0
    assert detail is not None
    assert detail.get("retry_used") is True
    assert model.temperatures[1] == 0.0


def test_history_truncation_guard_aborts_without_flag():
    step_cfg = {"history_keep_last_pairs": 3}
    assert history_truncation_active(step_cfg)
    with pytest.raises(SystemExit) as exc:
        enforce_full_history_or_exit(
            step_cfg,
            allow_history_truncation=False,
            script_name="test",
        )
    assert exc.value.code == 2


def test_history_truncation_allowed_with_flag():
    step_cfg = {"history_max_obs_chars": 500}
    assert enforce_full_history_or_exit(
        step_cfg,
        allow_history_truncation=True,
        script_name="test",
    )


def test_compare_signal_calibration_dual_collapse_policies():
    episodes = [
        {
            "steps_detail": [
                {
                    "step_index": 0,
                    "tle": {"mean_entropy": 0.2},
                    "vc": 80.0,
                    "correctness": "optimal",
                },
                {"step_index": 1, "tle": {"mean_entropy": 0.9}, "vc": 40.0, "correctness": "legal"},
            ]
        }
    ]
    out = compare_signal_calibration(episodes)
    assert "optimal_only" in out
    assert "legal_or_optimal" in out
    assert out["optimal_only"]["tle"]["collapse_policy"] == "optimal_only"
    assert out["legal_or_optimal"]["vc"]["collapse_policy"] == "legal_or_optimal"
    assert out["optimal_only"]["tle"]["n_steps"] == 2
    assert out["legal_or_optimal"]["vc"]["n_steps"] == 2
    assert out["optimal_only"]["vc"]["mean_signal_correct"] == 80.0
    assert out["legal_or_optimal"]["vc"]["mean_signal_correct"] == 60.0


@pytest.mark.parametrize("strategy", ["adaptive_tle", "adaptive_vc", "eager_style"])
def test_run_phase2_hard_fails_without_policy_artifact(tmp_path: Path, strategy: str):
    cfg = tmp_path / "phase2_no_policy.yaml"
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    cfg.write_text(
        f"""
phase2:
  strategies: [{strategy}]
  domains: [tower_of_hanoi]
  instances_per_domain: 1
  runs_per_condition: 1
episode:
  max_steps_per_episode: 1
model:
  name: mock
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "experiment" / "run_phase2.py"),
            "--config",
            str(cfg),
            "--checkpoint-dir",
            str(ckpt),
            "--no-timestamp-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    combined = (result.stderr or "") + (result.stdout or "")
    assert "policy_artifact" in combined.lower()
