"""resolve_step_fn_kwargs from YAML."""

from __future__ import annotations

from src.agent.compute_stages import DEFAULT_VC_FOLLOWUP_INSTRUCTION
from src.utils.step_config import resolve_step_fn_kwargs


def test_tower_of_hanoi_gets_default_prefix_and_action_cap():
    cfg = {
        "inference": {"max_tokens": 256, "temperature": 0.3},
        "vc": {"mode": "followup"},
        "domain_prompts": {"tower_of_hanoi": {"prefix": "Tower of Hanoi", "action_max_tokens": 32}},
    }
    k = resolve_step_fn_kwargs(cfg, "tower_of_hanoi")
    assert k["vc_mode"] == "followup"
    assert "Tower of Hanoi" in k["prompt_prefix"]
    assert k["action_max_tokens"] == 32
    assert k["followup_max_context_chars"] is None
    assert k["followup_cot_max_chars"] == 12000
    assert k["vc_raw_completion_max_chars"] == 8000
    assert k["vc_followup_instruction"] == DEFAULT_VC_FOLLOWUP_INSTRUCTION
    assert k["c1_cot_temperature"] is None
    assert k["c2_sample_temperature"] == 0.7


def test_vc_followup_instruction_from_yaml():
    custom = "Custom VC instruction.\n\nConfidence:"
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {"mode": "followup", "followup_instruction": custom},
        "domain_prompts": {"tower_of_hanoi": {"prefix": "x"}},
    }
    k = resolve_step_fn_kwargs(cfg, "tower_of_hanoi")
    assert k["vc_followup_instruction"] == custom


def test_vc_followup_budget_keys_from_yaml():
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {
            "mode": "followup",
            "followup_max_context_chars": 50000,
            "followup_cot_max_chars": 9000,
            "vc_raw_completion_max_chars": 6000,
        },
        "domain_prompts": {"tower_of_hanoi": {"prefix": "x"}},
    }
    k = resolve_step_fn_kwargs(cfg, "tower_of_hanoi")
    assert k["followup_max_context_chars"] == 50000
    assert k["followup_cot_max_chars"] == 9000
    assert k["vc_raw_completion_max_chars"] == 6000


def test_textworld_uses_default_action_cap_and_stop():
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {"mode": "inline"},
        "domain_prompts": {
            "textworld": {
                "prefix": "You are playing a text adventure.",
                "action_max_tokens": 32,
                "action_stop": ["\n"],
            }
        },
    }
    k = resolve_step_fn_kwargs(cfg, "textworld")
    assert k["action_max_tokens"] == 32
    assert k["action_temperature"] == 0.2
    assert k["action_stop"] == ["\n"]
    assert "text adventure" in k["prompt_prefix"].lower()


def test_textworld_action_max_tokens_override():
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {"mode": "inline"},
        "domain_prompts": {"textworld": {"prefix": "x", "action_max_tokens": 64}},
    }
    k = resolve_step_fn_kwargs(cfg, "textworld")
    assert k["action_max_tokens"] == 64


def test_textworld_disable_action_stop():
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {"mode": "inline", "textworld_action_stop": []},
    }
    k = resolve_step_fn_kwargs(cfg, "textworld")
    assert k["action_stop"] is None


def test_domain_cot_max_tokens_overrides_global_c1() -> None:
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {"mode": "inline"},
        "c1": {"cot_max_tokens": 1024},
        "domain_prompts": {
            "textworld": {"prefix": "tw", "cot_max_tokens": 512},
            "tower_of_hanoi": {"prefix": "toh", "cot_max_tokens": 2048},
        },
    }
    assert resolve_step_fn_kwargs(cfg, "textworld")["c1_cot_max_tokens"] == 512
    assert resolve_step_fn_kwargs(cfg, "tower_of_hanoi")["c1_cot_max_tokens"] == 2048


def test_domain_without_cot_max_tokens_uses_global_c1() -> None:
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {"mode": "inline"},
        "c1": {"cot_max_tokens": 1024},
        "domain_prompts": {"tower_of_hanoi": {"prefix": "x"}},
    }
    assert resolve_step_fn_kwargs(cfg, "tower_of_hanoi")["c1_cot_max_tokens"] == 1024


def test_c1_and_c2_knobs_from_yaml() -> None:
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {"mode": "inline"},
        "domain_prompts": {"tower_of_hanoi": {"prefix": "x"}},
        "c1": {
            "cot_temperature": 0.7,
            "cot_max_tokens": 222,
        },
        "c2": {"n_samples": 5, "sample_temperature": 0.65},
    }
    k = resolve_step_fn_kwargs(cfg, "tower_of_hanoi")
    assert k["c1_cot_temperature"] == 0.7
    assert k["c1_cot_max_tokens"] == 222
    assert k["c2_n_samples"] == 5
    assert k["c2_sample_temperature"] == 0.65
