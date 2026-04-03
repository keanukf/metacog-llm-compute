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


def test_textworld_uses_inference_max_tokens():
    cfg = {
        "inference": {"max_tokens": 128, "temperature": 0.2},
        "vc": {"mode": "inline"},
    }
    k = vc_step_fn_kwargs(cfg, "textworld")
    assert k["action_max_tokens"] == 128
    assert k["action_temperature"] == 0.2
