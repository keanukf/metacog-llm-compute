"""vc_step_fn_kwargs from YAML."""
from __future__ import annotations

from src.utils.vc_config import vc_step_fn_kwargs


def test_tower_of_hanoi_gets_default_prefix_and_action_cap():
    cfg = {
        "inference": {"max_tokens": 256, "temperature": 0.3},
        "vc": {"mode": "followup", "prompt_prefix": {}},
    }
    k = vc_step_fn_kwargs(cfg, "tower_of_hanoi")
    assert k["vc_mode"] == "followup"
    assert "Tower of Hanoi" in k["prompt_prefix"]
    assert k["action_max_tokens"] == 32


def test_textworld_uses_default_action_cap_and_stop():
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {"mode": "inline"},
    }
    k = vc_step_fn_kwargs(cfg, "textworld")
    assert k["action_max_tokens"] == 32
    assert k["action_temperature"] == 0.2
    assert k["action_stop"] == ["\n"]
    assert "text adventure" in k["prompt_prefix"].lower()


def test_textworld_action_max_tokens_override():
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {"mode": "inline", "action_max_tokens": {"textworld": 64}},
    }
    k = vc_step_fn_kwargs(cfg, "textworld")
    assert k["action_max_tokens"] == 64


def test_textworld_disable_action_stop():
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {"mode": "inline", "textworld_action_stop": []},
    }
    k = vc_step_fn_kwargs(cfg, "textworld")
    assert k["action_stop"] is None
