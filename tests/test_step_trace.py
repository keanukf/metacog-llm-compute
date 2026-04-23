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


class _NEnv:
    """Deterministic env with programmable observations; terminates after len(obs_seq)-1 steps."""

    def __init__(self, obs_seq: list[str]) -> None:
        self._obs_seq = list(obs_seq)
        self._i = 0
        self.done = False
        self.task_success = False
        self.step_results: list[dict] = []

    def reset(self) -> str:
        self._i = 0
        self.done = False
        return self._obs_seq[0]

    def step(self, action: str) -> str:
        self.step_results.append({"step_index": self._i, "correctness": "legal", "action_parsed": action})
        self._i += 1
        if self._i >= len(self._obs_seq) - 1:
            self.done = True
        return self._obs_seq[self._i]


def test_history_compaction_keeps_whole_pairs() -> None:
    env = _NEnv(["reset", "o1", "o2", "o3", "o4", "o5", "o6"])

    class _Model:
        def generate(self, prompt, logprobs=False, **kwargs):
            return "noop", ([{"logprob": -0.1}] if logprobs else None)

    seen_histories: list[list[str]] = []

    def step_fn(obs: str, history: list[str], model) -> tuple[str, None, None, int, int]:
        seen_histories.append(list(history))
        return "noop", None, None, 0, 1

    from src.agent.base_agent import run_episode

    run_episode(
        env,
        _Model(),
        "C0",
        step_fn=step_fn,
        max_steps=20,
        history_keep_last_pairs=2,
    )
    # By step 3+, compaction must apply: 1 reset + last 2 pairs => 1 + 4 lines = 5.
    assert len(seen_histories) >= 4
    h = seen_histories[-1]
    assert len(h) == 5
    assert h[0].startswith("OBSERVATION:")
    assert h[1].startswith("ACTION:")
    assert h[2].startswith("OBSERVATION:")
    assert h[3].startswith("ACTION:")
    assert h[4].startswith("OBSERVATION:")


def test_pinned_recipe_is_injected_after_discovery() -> None:
    recipe_obs = (
        "You open the cookbook.\n\n"
        "Recipe #1\n"
        "---------\n"
        "Ingredients:\nlettuce\nred apple\n\nDirections:\nchop the lettuce\n"
        "\n\n> Kitchen"
    )
    env = _NEnv(["reset", recipe_obs, "after", "after2", "after3"])

    class _Model:
        def generate(self, prompt, logprobs=False, **kwargs):
            return "noop", ([{"logprob": -0.1}] if logprobs else None)

    pinned_seen: list[bool] = []

    def step_fn(obs: str, history: list[str], model) -> tuple[str, None, None, int, int]:
        pinned_seen.append(any(isinstance(x, str) and x.startswith("PINNED RECIPE:") for x in history))
        return "noop", None, None, 0, 1

    from src.agent.base_agent import run_episode

    run_episode(
        env,
        _Model(),
        "C0",
        step_fn=step_fn,
        max_steps=10,
        history_keep_last_pairs=2,
        pin_recipe=True,
    )
    # First step (before recipe discovered) no pin; subsequent steps should have it.
    assert pinned_seen[0] is False
    assert any(pinned_seen[1:])


def test_null_trace_hook_accepts_new_kwargs() -> None:
    from src.utils.tracing import NullTraceHook

    h = NullTraceHook()
    h.episode_start("ep1", metadata={"a": 1}, session_id="s", tags=["t"], trace_name="n")
    h.log_step(
        step_index=0,
        stage="C0",
        observation="o",
        action="a",
        prompt="p",
        action_output="out",
        model_name="m",
        metadata={"x": 1},
        vc_prompt="vp",
        vc_output="vo",
    )
    h.episode_end(output={"ok": True}, final_tags=["done"])


def test_langfuse_trace_hook_prefers_update_trace_when_available() -> None:
    """If the SDK exposes update_trace(trace_id=...), we should use it for trace-level tags."""
    from src.utils.tracing import LangfuseTraceHook

    class _FakeSpan:
        def __init__(self) -> None:
            self.id = "span_1"

        def end(self) -> None:
            return None

    class _FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def create_trace_id(self) -> str:
            return "trace_1"

        def start_observation(self, **kwargs):
            # We only need a span-like object that supports id/end.
            return _FakeSpan()

        def update_trace(self, trace_id: str, **payload) -> None:
            self.calls.append(("update_trace", {"trace_id": trace_id, **payload}))

        def flush(self) -> None:
            return None

    c = _FakeClient()
    h = LangfuseTraceHook(c)
    h.episode_start("epX", session_id="sess", tags=["pilot", "textworld"], trace_name="epX")
    assert any(
        name == "update_trace" and call.get("trace_id") == "trace_1" and call.get("tags") == ["pilot", "textworld"]
        for name, call in c.calls
    )
