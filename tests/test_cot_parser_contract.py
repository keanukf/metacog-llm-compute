from __future__ import annotations

from src.agent.cot_parser import parse_cot_action


def test_parse_cot_action_post_think_contract() -> None:
    out = parse_cot_action("<think>reasoning</think>\nA->C")
    assert out["status"] == "parsed"
    assert out["parse_method"] == "post_think"
    assert out["action"] == "A->C"
    assert out["reasoning_internal"] == "reasoning"


def test_parse_cot_action_legacy_action_prefix_contract() -> None:
    out = parse_cot_action("ACTION: go north")
    assert out["status"] == "parsed"
    assert out["parse_method"] == "legacy_action_prefix"
    assert out["action"] == "go north"


def test_parse_cot_action_lmstudio_command_tag_contract() -> None:
    out = parse_cot_action("<think>plan</think>\n<reason>ok</reason>\n<command>go north</command>")
    assert out["action"] == "go north"
    assert out["parse_method"] == "lmstudio_command_tag"


def test_parse_cot_action_unparsed_contract() -> None:
    out = parse_cot_action("<think>only reasoning</think>")
    assert out["status"] == "unparsed"
    assert out["parse_method"] == "none"
    assert out["action"] == ""
