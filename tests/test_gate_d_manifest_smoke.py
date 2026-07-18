"""Regression test: gate_d_manifest_smoke.py called create_execution_backend with a
positional use_real argument, but the factory declares it keyword-only (``*, use_real``) —
every other caller in the codebase already used the keyword form. Never caught because
this script had never been run before (first run: 2026-07-18, Gate D TextWorld freeze)."""

from __future__ import annotations

from typing import Any

import src.execution.backend.factory as backend_factory
import src.utils.experiment_env as experiment_env
from scripts.gate_d_manifest_smoke import _run_domain_smoke


class _StubModel:
    def generate(self, prompt, logprobs=False, **kwargs):
        return "look", None


def test_run_domain_smoke_calls_create_execution_backend_with_keyword_arg(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_create_execution_backend(config, *, use_real):
        calls.append({"use_real": use_real})
        return _StubModel()

    def fake_run_episode(env, model, stage, *, step_fn, max_steps, **history_cfg):
        return {"task_success": False, "episode_length_steps": max_steps, "step_correctness": []}

    monkeypatch.setattr(backend_factory, "create_execution_backend", fake_create_execution_backend)
    monkeypatch.setattr(experiment_env, "make_experiment_env", lambda *a, **k: object())
    monkeypatch.setattr(
        "src.utils.manifest.load_manifest",
        lambda domain, config, root: {0: {"holdout": True, "difficulty_tier": "easy"}},
    )
    monkeypatch.setattr("src.agent.base_agent.run_episode", fake_run_episode)
    monkeypatch.setattr("src.utils.step_config.resolve_step_fn_kwargs", lambda config, domain: {})
    monkeypatch.setattr("src.agent.compute_stages.get_step_fn", lambda stage, **kwargs: None)

    report = _run_domain_smoke(
        domain="textworld",
        config={},
        production_cap=45,
        use_real_model=False,
    )

    assert calls == [{"use_real": False}]
    assert report["num_instances"] == 1
    assert report["instances"][0]["holdout"] is True
    assert report["instances"][0]["difficulty_tier"] == "easy"
