"""Step trace JSONL and extended step result normalization."""
from __future__ import annotations

import json
from pathlib import Path

from src.agent.base_agent import _normalize_step_result, run_episode
from src.agent.compute_stages import get_step_fn
from src.utils.logging_utils import write_step_trace_line


def test_normalize_step_result_nine_tuple():
    r = ("go south", {"mean_entropy": 0.1}, 50.0, 3, 2, None, None, "prompt text", "full\nresponse")
    a, tle, vc, tok, calls, lp, vd, p, resp = _normalize_step_result(r)
    assert a == "go south"
    assert tle == {"mean_entropy": 0.1}
    assert vc == 50.0
    assert tok == 3
    assert calls == 2
    assert p == "prompt text"
    assert resp == "full\nresponse"


def test_write_step_trace_line_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "trace_ep_x.jsonl"
    write_step_trace_line(p, {"a": 1, "b": "x"})
    write_step_trace_line(p, {"a": 2})
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1, "b": "x"}
    assert json.loads(lines[1]) == {"a": 2}


class _TwoStepEnv:
    def __init__(self) -> None:
        self._i = 0
        self.done = False
        self.task_success = False
        self.step_results: list[dict] = []

    def reset(self) -> str:
        self._i = 0
        self.done = False
        return "room A"

    def step(self, action: str) -> str:
        self._i += 1
        self.step_results.append(
            {"step_index": self._i - 1, "correctness": "legal", "action_parsed": action}
        )
        if self._i >= 2:
            self.done = True
        return f"after {action}"


class _EchoModel:
    def generate(self, prompt, logprobs=False, **kwargs):
        return "act_one_line\nextra", ([{"logprob": -0.1}] if logprobs else None)


def test_run_episode_writes_step_trace_jsonl(tmp_path: Path) -> None:
    env = _TwoStepEnv()
    model = _EchoModel()
    step_fn = get_step_fn("C0", vc_mode="none", prompt_prefix="P:")
    ep = "ep_test_trace"
    result = run_episode(
        env,
        model,
        "C0",
        step_fn=step_fn,
        max_steps=5,
        save_step_traces=True,
        episode_id=ep,
        trace_output_dir=str(tmp_path),
        trace_model_name="test-model",
        trace_hook=None,
    )
    assert result["steps"] == 2
    tfile = tmp_path / f"trace_{ep}.jsonl"
    assert tfile.is_file()
    lines = tfile.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["step_index"] == 0
    assert "P:" in row0["prompt_full"]
    assert "room A" in row0["prompt_full"]
    assert row0["response_full"] == "act_one_line\nextra"
    assert row0["action_parsed"] == "act_one_line"
    assert row0["observation_before"] == "room A"
    assert "ACTION:" not in row0["history_snapshot"]  # no prior actions yet
    assert "OBSERVATION: room A" in "\n".join(row0["history_snapshot"])  # reset() seeded into history
    row1 = json.loads(lines[1])
    assert row1["step_index"] == 1
    assert any(isinstance(x, str) and x.startswith("ACTION:") for x in row1["history_snapshot"])
    assert any(isinstance(x, str) and x.startswith("OBSERVATION:") for x in row1["history_snapshot"])


def test_run_episode_history_includes_prior_action(tmp_path: Path) -> None:
    """Second-step prompt must include ACTION: line from the first step."""
    env = _TwoStepEnv()
    model = _EchoModel()
    step_fn = get_step_fn("C0", vc_mode="none", prompt_prefix="TASK")
    run_episode(
        env,
        model,
        "C0",
        step_fn=step_fn,
        max_steps=5,
        save_step_traces=True,
        episode_id="ep_hist",
        trace_output_dir=str(tmp_path),
    )
    lines = (tmp_path / "trace_ep_hist.jsonl").read_text(encoding="utf-8").strip().splitlines()
    row1 = json.loads(lines[1])
    joined = "\n".join(row1["history_snapshot"])
    assert "OBSERVATION: room A" in joined
    assert "ACTION: act_one_line" in joined
    assert "OBSERVATION: after act_one_line" in joined
