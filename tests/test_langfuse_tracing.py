"""Langfuse trace hook hierarchy and field mapping (mock SDK, no cloud)."""

from __future__ import annotations

from typing import Any

from src.utils.tracing import LangfuseTraceHook

_OBS_REGISTRY: list["_RecordedObs"] = []


class _RecordedObs:
    def __init__(
        self,
        *,
        name: str,
        as_type: str,
        parent: _RecordedObs | None,
        recorder: list[dict[str, Any]],
    ) -> None:
        self.id = f"obs_{len(recorder)}"
        self.name = name
        self.as_type = as_type
        self._parent = parent
        self._recorder = recorder
        self._updates: list[dict[str, Any]] = []
        self._scores: list[dict[str, Any]] = []
        recorder.append(
            {
                "id": self.id,
                "name": name,
                "as_type": as_type,
                "parent_id": parent.id if parent else None,
            }
        )
        _OBS_REGISTRY.append(self)

    def update(self, **kwargs: Any) -> None:
        self._updates.append(dict(kwargs))

    def end(self) -> None:
        return None

    def score(self, **kwargs: Any) -> None:
        self._scores.append(dict(kwargs))

    def start_as_current_observation(self, **kwargs: Any) -> "_ObsContext":
        return _ObsContext(self, kwargs)

    def start_observation(self, **kwargs: Any) -> _RecordedObs:
        return _RecordedObs(
            name=str(kwargs.get("name") or ""),
            as_type=str(kwargs.get("as_type") or "span"),
            parent=self,
            recorder=self._recorder,
        )


class _ObsContext:
    def __init__(self, parent: _RecordedObs, kwargs: dict[str, Any]) -> None:
        self._parent = parent
        self._kwargs = kwargs
        self._obs: _RecordedObs | None = None

    def __enter__(self) -> _RecordedObs:
        self._obs = _RecordedObs(
            name=str(self._kwargs.get("name") or ""),
            as_type=str(self._kwargs.get("as_type") or "span"),
            parent=self._parent,
            recorder=self._parent._recorder,
        )
        return self._obs

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeOtelClient:
    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self.trace_updates: list[dict[str, Any]] = []

    def start_as_current_observation(self, **kwargs: Any) -> _ObsContext:
        return _ObsContext(
            _RecordedObs(
                name=str(kwargs.get("name") or "root"),
                as_type=str(kwargs.get("as_type") or "span"),
                parent=None,
                recorder=self.observations,
            ),
            kwargs,
        )

    def start_observation(self, **kwargs: Any) -> _RecordedObs:
        return _RecordedObs(
            name=str(kwargs.get("name") or ""),
            as_type=str(kwargs.get("as_type") or "span"),
            parent=None,
            recorder=self.observations,
        )

    def flush(self) -> None:
        return None


def _parent_chain(observations: list[dict[str, Any]], obs_id: str) -> list[str]:
    by_id = {o["id"]: o for o in observations}
    names: list[str] = []
    cur: str | None = obs_id
    while cur:
        row = by_id.get(cur)
        if not row:
            break
        names.append(str(row["name"]))
        cur = row.get("parent_id")
    return list(reversed(names))


def _reset_registry() -> None:
    _OBS_REGISTRY.clear()


def test_c1_deep_chain_parent_ids() -> None:
    _reset_registry()
    client = _FakeOtelClient()
    hook = LangfuseTraceHook(client)

    step = _RecordedObs(name="step_0", as_type="span", parent=None, recorder=client.observations)
    subcalls = [
        {
            "kind": "reason",
            "prompt": "reason in",
            "response": "<think>cot out</think>\ngo north",
            "tokens_generated": 10,
            "temperature": 0.5,
            "max_tokens": 128,
            "enable_thinking": True,
        },
    ]
    hook.log_step_children(
        step_span=step,
        step_index=0,
        stage="C1",
        action="go north",
        prompt="merged",
        action_output="merged out",
        model_name="test-model",
        metadata={"tle": {"mean_entropy": 0.42}, "vc": 80.0},
        vc_prompt="vc in",
        vc_output="80",
        subcalls=subcalls,
    )

    vc_rows = [o for o in client.observations if "vc_followup" in o["name"]]
    reason_rows = [o for o in client.observations if o["name"] == "reason_0_C1"]
    assert reason_rows and vc_rows
    assert vc_rows[0]["parent_id"] == reason_rows[0]["id"]
    chain = _parent_chain(client.observations, vc_rows[0]["id"])
    assert chain[-2:] == ["reason_0_C1", "vc_followup_0"]
    assert chain[0] == "step_0"


def test_c1_reason_gets_usage_and_tle_score() -> None:
    _reset_registry()
    client = _FakeOtelClient()
    hook = LangfuseTraceHook(client)
    step = _RecordedObs(name="step_0", as_type="span", parent=None, recorder=client.observations)
    subcalls = [
        {
            "kind": "reason",
            "prompt": "a",
            "response": "<think>b</think>\nd",
            "tokens_generated": 2,
            "temperature": 0.5,
        },
    ]
    hook.log_step_children(
        step_span=step,
        step_index=0,
        stage="C1",
        action="d",
        prompt="p",
        action_output="o",
        model_name="m",
        metadata={"tle": {"mean_entropy": 0.9}},
        subcalls=subcalls,
    )
    reason_obs = next(o for o in _OBS_REGISTRY if o.name == "reason_0_C1")
    assert any(u.get("usage_details") == {"output": 2} for u in reason_obs._updates)
    assert any(
        s.get("name") == "tle_mean_entropy" and s.get("value") == 0.9 for s in reason_obs._scores
    )


def test_emit_step_c0_action_and_vc_under_step() -> None:
    _reset_registry()
    client = _FakeOtelClient()
    hook = LangfuseTraceHook(client)
    step = _RecordedObs(name="step_0", as_type="span", parent=None, recorder=client.observations)
    hook.log_step_children(
        step_span=step,
        step_index=0,
        stage="C0",
        action="go east",
        prompt="prompt",
        action_output="go east",
        model_name="m",
        metadata={"vc": 50.0},
        vc_prompt="vc p",
        vc_output="50",
        subcalls=None,
    )
    names = [o["name"] for o in client.observations]
    assert "action_0_C0" in names
    assert "vc_followup_0" in names
    vc_row = next(o for o in client.observations if o["name"] == "vc_followup_0")
    assert vc_row["parent_id"] == step.id
