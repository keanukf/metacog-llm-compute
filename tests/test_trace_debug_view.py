"""Tests for compact debug views built from step trace JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.logging_utils import write_step_trace_line
from src.utils.trace_debug_view import (
    build_run_debug_views,
    compact_step_from_trace,
    resolve_step_trace_flags,
    truncate_text,
)


def test_truncate_text_empty():
    assert truncate_text(None) is None
    assert truncate_text("") is None


def test_truncate_text_short():
    out = truncate_text("hello", head_chars=10, tail_chars=10)
    assert out is not None
    assert out["head"] == "hello"
    assert out["truncated"] is False
    assert out["length"] == 5


def test_truncate_text_long():
    text = "a" * 2000
    out = truncate_text(text, head_chars=10, tail_chars=5)
    assert out is not None
    assert out["truncated"] is True
    assert len(out["head"]) == 10
    assert len(out["tail"]) == 5
    assert out["length"] == 2000


def test_compact_step_c0():
    row = {
        "step_index": 0,
        "compute_stage": "C0",
        "prompt_full": "TASK\nOBS: room",
        "response_full": "go north\nextra",
        "action_parsed": "go north",
        "correctness": "legal",
        "step_wall_time_s": 0.5,
        "lm_calls": 1,
        "tokens_generated": 10,
    }
    step = compact_step_from_trace(row, head_chars=100, tail_chars=100)
    assert step["compute_stage"] == "C0"
    assert step["pipeline"]["primary"]["prompt"]["head"].startswith("TASK")
    assert step["action_executed"] == "go north"
    assert "cot" not in step["pipeline"] or step["pipeline"].get("cot") is None


def test_compact_step_c1():
    row = {
        "step_index": 1,
        "compute_stage": "C1",
        "prompt_full": "verify prompt full",
        "response_full": "verify response full",
        "action_parsed": "take key",
        "cot_prompt": "cot prompt here",
        "cot_response": "draft",
        "verify_prompt": "verify prompt here",
        "verify_response": "Just the command.",
        "vc_followup_prompt": "vc prompt",
        "vc_followup_response": "Confidence: 80",
        "call_detail": {
            "stage": "C1",
            "parse_method": "draft_action",
            "draft_action": "take key",
            "fallback_source": "draft_action",
            "subcalls": [
                {"kind": "cot", "prompt": "cot p", "response": "cot r", "temperature": 0.5},
                {
                    "kind": "verify",
                    "prompt": "ver p",
                    "response": "ver r",
                    "fallback_source": "draft_action",
                },
            ],
        },
        "tle": {"mean_entropy": 0.2},
        "vc": 80.0,
    }
    step = compact_step_from_trace(row, head_chars=50, tail_chars=50)
    assert step["pipeline"]["cot"]["prompt"]["head"].startswith("cot")
    assert step["pipeline"]["verify"]["parse"]["parse_method"] == "draft_action"
    assert step["pipeline"]["vc_followup"]["response"]["head"].startswith("Confidence")
    assert step["signals"]["tle"]["mean_entropy"] == 0.2


def test_compact_step_c2_samples():
    row = {
        "step_index": 0,
        "compute_stage": "C2",
        "prompt_full": "shared prompt",
        "response_full": "sample blocks",
        "action_parsed": "go east",
        "call_detail": {
            "stage": "C2",
            "n_samples": 2,
            "winner_index": 1,
            "vote_counts": {"go east": 2},
            "subcalls": [
                {
                    "kind": "sample",
                    "sample_index": 0,
                    "prompt": "p0",
                    "response": "r0",
                    "raw_first_line": "go east",
                    "is_winner": False,
                },
                {
                    "kind": "sample",
                    "sample_index": 1,
                    "prompt": "p1",
                    "response": "r1",
                    "raw_first_line": "go east",
                    "is_winner": True,
                },
            ],
        },
    }
    step = compact_step_from_trace(row)
    samples = step["pipeline"]["c2_samples"]
    assert len(samples) == 2
    assert samples[1]["gen"]["is_winner"] is True
    assert step["pipeline"]["c2_vote"]["winner_index"] == 1


def test_resolve_step_trace_flags_forces_traces():
    save, write, h, t = resolve_step_trace_flags(
        {"logging": {"write_debug_views": True, "save_step_traces": False}}
    )
    assert save is True
    assert write is True
    assert h == 800
    assert t == 800


def test_build_run_debug_views_integration(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace_ep_test_0.jsonl"
    write_step_trace_line(
        trace_path,
        {
            "step_index": 0,
            "compute_stage": "C0",
            "prompt_full": "BEGIN_PROMPT_MARKER room A",
            "response_full": "go north",
            "action_parsed": "go north",
            "observation_before": "room A",
            "observation_after": "room B",
        },
    )
    (tmp_path / "ep_test_0.json").write_text(
        json.dumps(
            {
                "episode_id": "ep_test_0",
                "domain": "textworld",
                "compute_stage": "C0",
                "task_success": True,
                "steps": 1,
            }
        ),
        encoding="utf-8",
    )

    summary = build_run_debug_views(tmp_path, head_chars=200, tail_chars=50)
    assert summary is not None
    assert summary["episodes_built"] == 1
    assert summary["total_steps"] == 1

    debug_dir = tmp_path / "debug_views"
    assert (debug_dir / "run_summary.json").is_file()
    ep_doc = json.loads((debug_dir / "episode_ep_test_0.json").read_text(encoding="utf-8"))
    assert ep_doc["episode_summary"]["domain"] == "textworld"
    head = ep_doc["steps"][0]["pipeline"]["primary"]["prompt"]["head"]
    assert "BEGIN_PROMPT_MARKER" in head
