"""
Optional observability hooks: Langfuse cloud traces (no-op when disabled or missing SDK).

Configure via ``tracing`` YAML block and/or environment variables:
``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, optional ``LANGFUSE_HOST``.

Requires ``langfuse`` (see optional dependency group ``tracing`` in pyproject.toml).
"""

from __future__ import annotations

import contextlib
import inspect
import os
import warnings
from typing import Any, Protocol


class TraceHook(Protocol):
    """Per-episode tracing (file logging is handled separately in base_agent)."""

    def episode_start(
        self,
        episode_id: str,
        metadata: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        tags: list[str] | None = None,
        trace_name: str | None = None,
    ) -> None: ...

    def log_action_generation(
        self,
        *,
        step_index: int,
        compute_stage: str,
        prompt: str,
        output: str,
        model_name: str | None,
        metadata: dict[str, Any] | None,
    ) -> None: ...

    def episode_end(
        self,
        *,
        output: dict[str, Any] | None = None,
        final_tags: list[str] | None = None,
    ) -> None: ...

    def start_step_observation(  # pragma: no cover - protocol only
        self,
        *,
        step_index: int,
        stage: str,
        observation: str,
        metadata: dict[str, Any] | None,
    ): ...

    def log_step_children(  # pragma: no cover - protocol only
        self,
        *,
        step_span: Any,
        step_index: int,
        stage: str,
        action: str,
        prompt: str,
        action_output: str,
        model_name: str | None,
        metadata: dict[str, Any] | None,
        strategy: str | None = None,
        vc_prompt: str | None = None,
        vc_output: str | None = None,
        subcalls: list[dict[str, Any]] | None = None,
    ) -> None: ...

    def log_step(  # pragma: no cover - protocol only
        self,
        *,
        step_index: int,
        stage: str,
        observation: str,
        action: str,
        prompt: str,
        action_output: str,
        model_name: str | None,
        metadata: dict[str, Any] | None,
        strategy: str | None = None,
        vc_prompt: str | None = None,
        vc_output: str | None = None,
        subcalls: list[dict[str, Any]] | None = None,
    ) -> None: ...


class NullTraceHook:
    """No-op tracer when Langfuse is off or unavailable."""

    def episode_start(
        self,
        episode_id: str,
        metadata: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        tags: list[str] | None = None,
        trace_name: str | None = None,
    ) -> None:
        return None

    def log_action_generation(
        self,
        *,
        step_index: int,
        compute_stage: str,
        prompt: str,
        output: str,
        model_name: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        return None

    def episode_end(
        self,
        *,
        output: dict[str, Any] | None = None,
        final_tags: list[str] | None = None,
    ) -> None:
        return None

    def log_step(
        self,
        *,
        step_index: int,
        stage: str,
        observation: str,
        action: str,
        prompt: str,
        action_output: str,
        model_name: str | None,
        metadata: dict[str, Any] | None,
        strategy: str | None = None,
        vc_prompt: str | None = None,
        vc_output: str | None = None,
        subcalls: list[dict[str, Any]] | None = None,
    ) -> None:
        return None

    def start_step_observation(
        self,
        *,
        step_index: int,
        stage: str,
        observation: str,
        metadata: dict[str, Any] | None,
    ):
        return contextlib.nullcontext(None)

    def log_step_children(
        self,
        *,
        step_span: Any,
        step_index: int,
        stage: str,
        action: str,
        prompt: str,
        action_output: str,
        model_name: str | None,
        metadata: dict[str, Any] | None,
        strategy: str | None = None,
        vc_prompt: str | None = None,
        vc_output: str | None = None,
        subcalls: list[dict[str, Any]] | None = None,
    ) -> None:
        return None


def _subcall_by_kind(subcalls: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for sc in subcalls:
        if isinstance(sc, dict) and str(sc.get("kind") or "").strip().lower() == kind:
            return sc
    return None


def _metadata_str(val: Any, *, max_len: int = 200) -> str:
    s = str(val)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _propagate_metadata_from_episode(
    episode_id: str,
    metadata: dict[str, Any] | None,
    *,
    trace_name: str | None,
) -> dict[str, str]:
    """String metadata safe for Langfuse ``propagate_attributes`` (≤200 chars per value)."""
    out: dict[str, str] = {"episode_id": _metadata_str(episode_id)}
    if trace_name:
        out["trace_name"] = _metadata_str(trace_name)
    meta = metadata or {}
    for key in ("compute_stage", "strategy", "model"):
        if meta.get(key) is not None:
            out[key] = _metadata_str(meta[key])
    return out


def _step_span_metadata(
    metadata: dict[str, Any] | None,
    *,
    stage: str,
    strategy: str | None = None,
    action: str | None = None,
) -> dict[str, Any]:
    """Step-level metadata: timing and stage only (scores live on observations)."""
    step_meta: dict[str, Any] = {"stage": stage}
    if action is not None:
        step_meta["action"] = action
    if strategy:
        step_meta["strategy"] = strategy
    base = metadata or {}
    for key in ("lm_calls", "step_wall_time_s", "step_start_time_utc", "tokens_generated"):
        if key in base and base[key] is not None:
            step_meta[key] = base[key]
    return step_meta


def _subcall_observation_metadata(
    stage: str,
    kind: str,
    sc: dict[str, Any],
    *,
    strategy: str | None = None,
    lm_call_index: int | None = None,
) -> dict[str, Any]:
    sc_meta: dict[str, Any] = {
        "stage": stage,
        "kind": kind,
        "counts_toward_compute": True,
    }
    if lm_call_index is not None:
        sc_meta["lm_call_index"] = lm_call_index
    if strategy:
        sc_meta["strategy"] = strategy
    if isinstance(sc.get("sample_index"), int):
        sc_meta["sample_index"] = int(sc.get("sample_index"))
    if isinstance(sc.get("is_winner"), bool):
        sc_meta["is_winner"] = bool(sc.get("is_winner"))
    if sc.get("fallback_source") is not None:
        sc_meta["fallback_source"] = sc.get("fallback_source")
    return sc_meta


def _usage_details_from_subcall(sc: dict[str, Any]) -> dict[str, int] | None:
    tok = sc.get("tokens_generated")
    if isinstance(tok, int) and tok >= 0:
        return {"output": tok}
    return None


def _model_parameters_from_subcall(sc: dict[str, Any]) -> dict[str, Any] | None:
    params: dict[str, Any] = {}
    if isinstance(sc.get("temperature"), (int, float)):
        params["temperature"] = float(sc.get("temperature"))
    if sc.get("max_tokens") is not None:
        params["max_tokens"] = sc.get("max_tokens")
    return params or None


def _tle_scalar(tle: Any) -> float | None:
    if tle is None:
        return None
    if isinstance(tle, (int, float)):
        return float(tle)
    if isinstance(tle, dict):
        for key in ("mean_entropy", "mean", "entropy"):
            val = tle.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    return None


class LangfuseTraceHook:
    """
    One Langfuse trace per episode (shared trace_id), nested observations per env step.
    Uses the Langfuse Python SDK v3+ API (``start_as_current_observation`` + OTel context).
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._trace_id: str | None = None
        self._root_span: Any = None
        self._episode_cm: Any = None
        self._propagate_cm: Any = None
        self._episode_tags: list[str] = []
        self._session_id: str | None = None
        self._trace_name: str | None = None
        self._supports_parent_observation_id = False
        self._supports_observation_tags = False
        self._supports_observation_times = False
        self._otel_api = False
        self._otel_has_start_as_current_observation = False
        self._otel_has_start_span = False
        try:
            sig = inspect.signature(getattr(self._client, "start_observation"))
            self._supports_parent_observation_id = "parent_observation_id" in sig.parameters
            self._supports_observation_tags = "tags" in sig.parameters
            self._supports_observation_times = ("start_time" in sig.parameters) or (
                "end_time" in sig.parameters
            )
        except Exception:
            pass

        self._otel_has_start_as_current_observation = callable(
            getattr(self._client, "start_as_current_observation", None)
        )
        self._otel_has_start_span = callable(getattr(self._client, "start_span", None)) or callable(
            getattr(self._client, "start_observation", None)
        )
        self._otel_api = bool(self._otel_has_start_as_current_observation) and bool(
            self._otel_has_start_span
        )

    @staticmethod
    def _parent_has_nested_current(parent: Any) -> bool:
        return callable(getattr(parent, "start_as_current_observation", None))

    def _end_observation(self, obs: Any, *, end_time: Any | None = None) -> None:
        if obs is None:
            return
        try:
            if end_time is not None:
                obs.end(end_time=end_time)
            else:
                obs.end()
        except TypeError:
            obs.end()

    def _update_generation(
        self,
        gen: Any,
        *,
        output: str,
        metadata: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> None:
        upd_kw: dict[str, Any] = {"output": output}
        if metadata:
            upd_kw["metadata"] = metadata
        if usage_details:
            upd_kw["usage_details"] = usage_details
        if model_parameters:
            upd_kw["model_parameters"] = model_parameters
        try:
            gen.update(**upd_kw)
        except TypeError:
            try:
                gen.update(output=output)
            except Exception:
                pass
        except Exception:
            pass

    def _score_observation(
        self,
        obs: Any,
        *,
        name: str,
        value: Any,
        data_type: str | None = None,
        comment: str | None = None,
    ) -> None:
        if value is None:
            return
        score_meth = getattr(obs, "score", None)
        if not callable(score_meth):
            return
        kw: dict[str, Any] = {"name": name, "value": value}
        if data_type:
            kw["data_type"] = data_type
        if comment:
            kw["comment"] = comment
        try:
            score_meth(**kw)
        except TypeError:
            try:
                score_meth(name, value)
            except Exception:
                pass
        except Exception:
            pass

    def _apply_verify_scores(self, obs: Any, metadata: dict[str, Any] | None) -> None:
        if not metadata:
            return
        tle_val = _tle_scalar(metadata.get("tle"))
        if tle_val is not None:
            self._score_observation(
                obs, name="tle_mean_entropy", value=tle_val, data_type="NUMERIC"
            )

    def _apply_step_scores(self, obs: Any, metadata: dict[str, Any] | None) -> None:
        if not metadata:
            return
        correctness = metadata.get("correctness")
        if correctness is not None:
            self._score_observation(
                obs,
                name="correctness",
                value=str(correctness),
                data_type="CATEGORICAL",
            )

    def _start_generation_child(
        self,
        *,
        parent_span: Any,
        ctx: Any,
        name: str,
        model: str,
        input_text: str,
        output_text: str,
        metadata: dict[str, Any],
        tags: list[str] | None,
        start_time: Any | None,
        end_time: Any | None,
        usage_details: dict[str, int] | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> Any:
        gen_kw: dict[str, Any] = {}
        if self._supports_observation_tags and tags:
            gen_kw["tags"] = list(tags)
        if self._supports_observation_times:
            if start_time is not None:
                gen_kw["start_time"] = start_time
            if end_time is not None:
                gen_kw["end_time"] = end_time

        parent_id = getattr(parent_span, "id", None)
        if parent_id is not None:
            try:
                gen = self._client.start_observation(
                    name=name,
                    as_type="generation",
                    trace_context=ctx,
                    parent_observation_id=parent_id,
                    model=model,
                    input=input_text,
                    metadata=metadata,
                    **gen_kw,
                )
                self._update_generation(
                    gen,
                    output=output_text,
                    usage_details=usage_details,
                    model_parameters=model_parameters,
                )
                return gen
            except TypeError:
                pass

        for meth_name in ("generation", "start_generation", "start_observation"):
            meth = getattr(parent_span, meth_name, None)
            if not callable(meth):
                continue
            try:
                if meth_name == "start_observation":
                    gen = meth(
                        name=name,
                        as_type="generation",
                        trace_context=ctx,
                        model=model,
                        input=input_text,
                        metadata=metadata,
                        **gen_kw,
                    )
                else:
                    gen = meth(
                        name=name,
                        model=model,
                        input=input_text,
                        metadata=metadata,
                        **gen_kw,
                    )
                self._update_generation(
                    gen,
                    output=output_text,
                    usage_details=usage_details,
                    model_parameters=model_parameters,
                )
                return gen
            except TypeError:
                continue
            except Exception:
                break

        gen = self._client.start_observation(
            name=name,
            as_type="generation",
            trace_context=ctx,
            model=model,
            input=input_text,
            metadata=metadata,
            **gen_kw,
        )
        self._update_generation(
            gen,
            output=output_text,
            usage_details=usage_details,
            model_parameters=model_parameters,
        )
        return gen

    def _log_subcall_generation_otel(
        self,
        parent: Any,
        *,
        name: str,
        model_name: str,
        sc: dict[str, Any],
        sc_meta: dict[str, Any],
    ) -> Any:
        """Create one generation under ``parent``; return the observation (context exited)."""
        usage = _usage_details_from_subcall(sc)
        model_params = _model_parameters_from_subcall(sc)
        inp = str(sc.get("prompt") or "")
        out = str(sc.get("response") or "")

        if self._parent_has_nested_current(parent):
            with parent.start_as_current_observation(
                as_type="generation",
                name=name,
                model=model_name,
                input=inp,
                metadata=sc_meta,
            ) as gen:
                self._update_generation(
                    gen,
                    output=out,
                    usage_details=usage,
                    model_parameters=model_params,
                )
                return gen

        gen = parent.start_observation(
            name=name,
            as_type="generation",
            model=model_name,
            input=inp,
            metadata=sc_meta,
        )
        self._update_generation(
            gen,
            output=out,
            usage_details=usage,
            model_parameters=model_params,
        )
        self._end_observation(gen)
        return gen

    def _log_vc_under_parent(
        self,
        parent: Any,
        *,
        step_index: int,
        stage: str,
        model_name: str,
        vc_prompt: str,
        vc_output: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        vc_meta: dict[str, Any] = {"stage": stage, "kind": "vc_followup"}
        if self._parent_has_nested_current(parent):
            with parent.start_as_current_observation(
                as_type="generation",
                name=f"vc_followup_{step_index}",
                model=model_name,
                input=vc_prompt,
                metadata=vc_meta,
            ) as vc_gen:
                self._update_generation(vc_gen, output=vc_output)
                vc_val = (metadata or {}).get("vc")
                if isinstance(vc_val, (int, float)):
                    self._score_observation(
                        vc_gen, name="vc", value=float(vc_val), data_type="NUMERIC"
                    )
            return

        vc_gen = parent.start_observation(
            as_type="generation",
            name=f"vc_followup_{step_index}",
            model=model_name,
            input=vc_prompt,
            metadata=vc_meta,
        )
        self._update_generation(vc_gen, output=vc_output)
        vc_val = (metadata or {}).get("vc")
        if isinstance(vc_val, (int, float)):
            self._score_observation(vc_gen, name="vc", value=float(vc_val), data_type="NUMERIC")
        self._end_observation(vc_gen)

    def _log_c1_chain(
        self,
        step_span: Any,
        *,
        step_index: int,
        stage: str,
        subcalls: list[dict[str, Any]],
        model_name: str,
        metadata: dict[str, Any] | None,
        strategy: str | None,
        vc_prompt: str | None,
        vc_output: str | None,
        ctx: Any | None = None,
    ) -> bool:
        """
        Log C1 as a single reason generation → optional vc_followup.
        Returns True if chain was logged.
        """
        reason_sc = _subcall_by_kind(subcalls, "reason") or _subcall_by_kind(subcalls, "cot")
        if reason_sc is None:
            return False

        model = model_name or "unknown"

        if self._parent_has_nested_current(step_span):
            with step_span.start_as_current_observation(
                as_type="generation",
                name=f"reason_{step_index}_{stage}",
                model=model,
                input=str(reason_sc.get("prompt") or ""),
                metadata=_subcall_observation_metadata(
                    stage, "reason", reason_sc, strategy=strategy, lm_call_index=1
                ),
            ) as reason_gen:
                self._update_generation(
                    reason_gen,
                    output=str(reason_sc.get("response") or ""),
                    usage_details=_usage_details_from_subcall(reason_sc),
                    model_parameters=_model_parameters_from_subcall(reason_sc),
                )
                self._apply_verify_scores(reason_gen, metadata)
                if vc_prompt and vc_output:
                    self._log_vc_under_parent(
                        reason_gen,
                        step_index=step_index,
                        stage=stage,
                        model_name=model,
                        vc_prompt=vc_prompt,
                        vc_output=vc_output,
                        metadata=metadata,
                    )
            return True

        if ctx is None:
            return False
        reason_gen = self._start_generation_child(
            parent_span=step_span,
            ctx=ctx,
            name=f"reason_{step_index}_{stage}",
            model=model,
            input_text=str(reason_sc.get("prompt") or ""),
            output_text=str(reason_sc.get("response") or ""),
            metadata=_subcall_observation_metadata(
                stage, "reason", reason_sc, strategy=strategy, lm_call_index=1
            ),
            tags=list(self._episode_tags) if self._episode_tags else None,
            start_time=None,
            end_time=None,
            usage_details=_usage_details_from_subcall(reason_sc),
            model_parameters=_model_parameters_from_subcall(reason_sc),
        )
        self._apply_verify_scores(reason_gen, metadata)
        self._end_observation(reason_gen)
        if vc_prompt and vc_output:
            vc_gen = self._start_generation_child(
                parent_span=reason_gen,
                ctx=ctx,
                name=f"vc_followup_{step_index}",
                model=model,
                input_text=vc_prompt,
                output_text=vc_output,
                metadata={"stage": stage, "kind": "vc_followup"},
                tags=list(self._episode_tags) if self._episode_tags else None,
                start_time=None,
                end_time=None,
            )
            vc_val = (metadata or {}).get("vc")
            if isinstance(vc_val, (int, float)):
                self._score_observation(vc_gen, name="vc", value=float(vc_val), data_type="NUMERIC")
            self._end_observation(vc_gen)
        return True

    def _log_c2_vote_aggregation(
        self,
        step_span: Any,
        *,
        step_index: int,
        stage: str,
        metadata: dict[str, Any] | None,
        strategy: str | None,
    ) -> None:
        """Log C2 majority-vote summary as a span sibling under the step."""
        cd = (metadata or {}).get("call_detail")
        if not isinstance(cd, dict) or str(cd.get("stage") or "") != "C2":
            return
        vote_meta: dict[str, Any] = {
            "stage": stage,
            "strategy": strategy,
            "winner_index": cd.get("winner_index"),
            "winning_vote_key": cd.get("winning_vote_key"),
            "tie_broken": cd.get("tie_broken"),
            "vote_counts": cd.get("vote_counts"),
            "vote_agreement": cd.get("vote_agreement"),
            "n_samples": cd.get("n_samples"),
            "sample_temperature": cd.get("sample_temperature"),
            "enable_thinking": cd.get("enable_thinking"),
        }
        subcalls = cd.get("subcalls")
        if isinstance(subcalls, list):
            vote_meta["per_sample_tle"] = [
                {
                    "sample_index": sc.get("sample_index"),
                    "tle": sc.get("tle"),
                    "is_winner": sc.get("is_winner"),
                }
                for sc in subcalls
                if isinstance(sc, dict)
            ]
        if self._parent_has_nested_current(step_span):
            with step_span.start_as_current_observation(
                as_type="span",
                name=f"vote_{step_index}_{stage}",
                metadata=vote_meta,
            ):
                pass
            return
        vote_span = step_span.start_observation(
            name=f"vote_{step_index}_{stage}",
            as_type="span",
            metadata=vote_meta,
        )
        self._end_observation(vote_span)

    def _log_c2_samples(
        self,
        step_span: Any,
        *,
        step_index: int,
        stage: str,
        subcalls: list[dict[str, Any]],
        model_name: str,
        strategy: str | None,
        ctx: Any | None = None,
    ) -> bool:
        logged = False
        model = model_name or "unknown"
        for sc in subcalls:
            if not isinstance(sc, dict):
                continue
            if str(sc.get("kind") or "").strip().lower() != "sample":
                continue
            sc_meta = _subcall_observation_metadata(stage, "sample", sc, strategy=strategy)
            si = sc_meta.get("sample_index", 0)
            name = f"sample_{si}_{step_index}_{stage}"
            if ctx is not None and not self._parent_has_nested_current(step_span):
                gen = self._start_generation_child(
                    parent_span=step_span,
                    ctx=ctx,
                    name=name,
                    model=model,
                    input_text=str(sc.get("prompt") or ""),
                    output_text=str(sc.get("response") or ""),
                    metadata=sc_meta,
                    tags=list(self._episode_tags) if self._episode_tags else None,
                    start_time=None,
                    end_time=None,
                    usage_details=_usage_details_from_subcall(sc),
                    model_parameters=_model_parameters_from_subcall(sc),
                )
                self._end_observation(gen)
            else:
                self._log_subcall_generation_otel(
                    step_span,
                    name=name,
                    model_name=model,
                    sc=sc,
                    sc_meta=sc_meta,
                )
            logged = True
        return logged

    def _log_default_action(
        self,
        step_span: Any,
        *,
        step_index: int,
        stage: str,
        action: str,
        prompt: str,
        action_output: str,
        model_name: str,
        metadata: dict[str, Any] | None,
        strategy: str | None,
        ctx: Any | None = None,
    ) -> Any | None:
        """Log a single action generation (C0 etc.). Returns open gen when manual end needed."""
        action_meta = _step_span_metadata(metadata, stage=stage, strategy=strategy, action=action)
        model = model_name or "unknown"

        if ctx is not None and not self._parent_has_nested_current(step_span):
            gen = self._start_generation_child(
                parent_span=step_span,
                ctx=ctx,
                name=f"action_{step_index}_{stage}",
                model=model,
                input_text=prompt,
                output_text=action_output,
                metadata=action_meta,
                tags=list(self._episode_tags) if self._episode_tags else None,
                start_time=None,
                end_time=None,
            )
            self._apply_verify_scores(gen, metadata)
            self._end_observation(gen)
            return None

        if self._parent_has_nested_current(step_span):
            gen = step_span.start_observation(
                name=f"action_{step_index}_{stage}",
                as_type="generation",
                model=model,
                input=prompt,
                metadata=action_meta,
            )
            self._update_generation(gen, output=action_output)
            self._apply_verify_scores(gen, metadata)
            return gen

        gen = step_span.start_observation(
            name=f"action_{step_index}_{stage}",
            as_type="generation",
            model=model,
            input=prompt,
            metadata=action_meta,
        )
        self._update_generation(gen, output=action_output)
        self._apply_verify_scores(gen, metadata)
        self._end_observation(gen)
        return None

    def _emit_step_children(
        self,
        step_span: Any,
        *,
        step_index: int,
        stage: str,
        action: str,
        prompt: str,
        action_output: str,
        model_name: str | None,
        metadata: dict[str, Any] | None,
        strategy: str | None = None,
        vc_prompt: str | None = None,
        vc_output: str | None = None,
        subcalls: list[dict[str, Any]] | None = None,
        ctx: Any | None = None,
    ) -> None:
        did_subcalls = False
        if stage == "C1" and isinstance(subcalls, list) and subcalls:
            did_subcalls = self._log_c1_chain(
                step_span,
                step_index=step_index,
                stage=stage,
                subcalls=subcalls,
                model_name=model_name or "unknown",
                metadata=metadata,
                strategy=strategy,
                vc_prompt=vc_prompt,
                vc_output=vc_output,
                ctx=ctx,
            )

        if stage == "C2" and isinstance(subcalls, list) and subcalls:
            if self._log_c2_samples(
                step_span,
                step_index=step_index,
                stage=stage,
                subcalls=subcalls,
                model_name=model_name or "unknown",
                strategy=strategy,
                ctx=ctx,
            ):
                did_subcalls = True
            self._log_c2_vote_aggregation(
                step_span,
                step_index=step_index,
                stage=stage,
                metadata=metadata,
                strategy=strategy,
            )

        action_gen: Any | None = None
        if not did_subcalls:
            action_gen = self._log_default_action(
                step_span,
                step_index=step_index,
                stage=stage,
                action=action,
                prompt=prompt,
                action_output=action_output,
                model_name=model_name or "unknown",
                metadata=metadata,
                strategy=strategy,
                ctx=ctx,
            )

        # C0/C2: VC is sibling under step when not already logged under C1 reason gen.
        if (not did_subcalls) and vc_prompt and vc_output:
            self._log_vc_under_parent(
                step_span,
                step_index=step_index,
                stage=stage,
                model_name=model_name or "unknown",
                vc_prompt=vc_prompt,
                vc_output=vc_output,
                metadata=metadata,
            )

        if action_gen is not None:
            self._end_observation(action_gen)

    def episode_start(
        self,
        episode_id: str,
        metadata: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        tags: list[str] | None = None,
        trace_name: str | None = None,
    ) -> None:
        self._trace_id = None
        self._root_span = None
        self._episode_cm = None
        self._propagate_cm = None
        self._episode_tags = list(tags or [])
        self._session_id = session_id
        self._trace_name = trace_name
        meta = dict(metadata or {})
        meta["episode_id"] = episode_id
        try:
            if self._otel_api:
                root_name = str(trace_name) if trace_name else "metacog_episode"
                prop_meta = _propagate_metadata_from_episode(
                    episode_id, meta, trace_name=trace_name
                )
                try:
                    from langfuse import propagate_attributes  # type: ignore[import-not-found]

                    self._propagate_cm = propagate_attributes(metadata=prop_meta)
                    self._propagate_cm.__enter__()
                except ImportError:
                    self._propagate_cm = None
                except Exception:
                    self._propagate_cm = None

                self._episode_cm = self._client.start_as_current_observation(
                    as_type="span",
                    name=root_name,
                    metadata=prop_meta,
                )
                self._root_span = self._episode_cm.__enter__()
                self._trace_id = getattr(self._root_span, "trace_id", None)

                payload: dict[str, Any] = {}
                if trace_name:
                    payload["name"] = str(trace_name)
                if session_id:
                    payload["session_id"] = str(session_id)
                if self._episode_tags:
                    payload["tags"] = list(self._episode_tags)
                if payload:
                    upd_trace = getattr(self._root_span, "update_trace", None)
                    if callable(upd_trace):
                        try:
                            upd_trace(**payload)
                        except Exception:
                            pass
                    elif self._trace_id is not None:
                        client_upd = getattr(self._client, "update_trace", None)
                        if callable(client_upd):
                            try:
                                client_upd(trace_id=self._trace_id, **payload)
                            except TypeError:
                                try:
                                    client_upd(id=self._trace_id, **payload)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                return

            self._trace_id = self._client.create_trace_id()
            obs_kw: dict[str, Any] = {
                "name": "metacog_episode",
                "as_type": "span",
                "metadata": meta,
            }
            try:
                from langfuse.types import TraceContext  # type: ignore[import-not-found]

                obs_kw["trace_context"] = TraceContext(trace_id=self._trace_id)
            except ImportError:
                pass
            self._root_span = self._client.start_observation(**obs_kw)
            trace_fields: dict[str, Any] = {}
            if trace_name:
                trace_fields["name"] = str(trace_name)
            if session_id:
                trace_fields["session_id"] = str(session_id)
            if self._episode_tags:
                trace_fields["tags"] = list(self._episode_tags)
            if trace_fields:
                upd_trace = getattr(self._client, "update_trace", None)
                if callable(upd_trace):
                    try:
                        upd_trace(trace_id=self._trace_id, **trace_fields)
                    except TypeError:
                        try:
                            upd_trace(id=self._trace_id, **trace_fields)
                        except Exception as e:
                            warnings.warn(f"Langfuse update_trace failed: {e!s}")
                    except Exception as e:
                        warnings.warn(f"Langfuse update_trace failed: {e!s}")
        except Exception as e:
            warnings.warn(f"Langfuse episode_start failed: {e!s}")
            self._trace_id = None
            self._root_span = None

    def log_action_generation(
        self,
        *,
        step_index: int,
        compute_stage: str,
        prompt: str,
        output: str,
        model_name: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        if self._trace_id is None:
            return
        try:
            from langfuse.types import TraceContext

            ctx = TraceContext(trace_id=self._trace_id)
            gen_kw: dict[str, Any] = {}
            if self._supports_observation_tags and self._episode_tags:
                gen_kw["tags"] = list(self._episode_tags)
            gen = self._client.start_observation(
                name=f"step_{step_index}_{compute_stage}",
                as_type="generation",
                trace_context=ctx,
                model=model_name or "unknown",
                input=prompt,
                output=output,
                metadata=metadata or {},
                **gen_kw,
            )
            gen.end()
        except Exception as e:
            warnings.warn(
                f"Langfuse log_action_generation failed (step_{step_index}_{compute_stage}): {e!s}"
            )

    def start_step_observation(
        self,
        *,
        step_index: int,
        stage: str,
        observation: str,
        metadata: dict[str, Any] | None,
    ):
        if self._otel_api and self._root_span is not None:
            step_meta = _step_span_metadata(metadata, stage=stage)
            return self._client.start_as_current_observation(
                as_type="span",
                name=f"step_{step_index}",
                input=observation,
                metadata=step_meta,
            )
        return contextlib.nullcontext(None)

    def log_step_children(
        self,
        *,
        step_span: Any,
        step_index: int,
        stage: str,
        action: str,
        prompt: str,
        action_output: str,
        model_name: str | None,
        metadata: dict[str, Any] | None,
        strategy: str | None = None,
        vc_prompt: str | None = None,
        vc_output: str | None = None,
        subcalls: list[dict[str, Any]] | None = None,
    ) -> None:
        if not (self._otel_api and step_span is not None):
            return
        try:
            self._emit_step_children(
                step_span,
                step_index=step_index,
                stage=stage,
                action=action,
                prompt=prompt,
                action_output=action_output,
                model_name=model_name,
                metadata=metadata,
                strategy=strategy,
                vc_prompt=vc_prompt,
                vc_output=vc_output,
                subcalls=subcalls,
            )
            self._apply_step_scores(step_span, metadata)
        except Exception as e:
            warnings.warn(f"Langfuse log_step_children failed (step_{step_index}): {e!s}")

    def log_step(
        self,
        *,
        step_index: int,
        stage: str,
        observation: str,
        action: str,
        prompt: str,
        action_output: str,
        model_name: str | None,
        metadata: dict[str, Any] | None,
        strategy: str | None = None,
        vc_prompt: str | None = None,
        vc_output: str | None = None,
        subcalls: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._trace_id is None and self._root_span is None:
            return
        try:
            if self._otel_api and self._root_span is not None:
                step_meta = _step_span_metadata(
                    metadata, stage=stage, strategy=strategy, action=action
                )
                with self._client.start_as_current_observation(
                    as_type="span",
                    name=f"step_{step_index}",
                    input=observation,
                    metadata=step_meta,
                ) as step_span:
                    self._emit_step_children(
                        step_span,
                        step_index=step_index,
                        stage=stage,
                        action=action,
                        prompt=prompt,
                        action_output=action_output,
                        model_name=model_name,
                        metadata=metadata,
                        strategy=strategy,
                        vc_prompt=vc_prompt,
                        vc_output=vc_output,
                        subcalls=subcalls,
                    )
                    self._apply_step_scores(step_span, metadata)
                return

            from langfuse.types import TraceContext

            ctx = TraceContext(trace_id=self._trace_id)
            step_meta = _step_span_metadata(metadata, stage=stage, strategy=strategy, action=action)
            step_kw: dict[str, Any] = {}
            if self._supports_observation_tags and self._episode_tags:
                step_kw["tags"] = list(self._episode_tags)
            step_span = self._client.start_observation(
                name=f"step_{step_index}",
                as_type="span",
                trace_context=ctx,
                input=observation,
                metadata=step_meta,
                **step_kw,
            )
            self._emit_step_children(
                step_span,
                step_index=step_index,
                stage=stage,
                action=action,
                prompt=prompt,
                action_output=action_output,
                model_name=model_name,
                metadata=metadata,
                strategy=strategy,
                vc_prompt=vc_prompt,
                vc_output=vc_output,
                subcalls=subcalls,
                ctx=ctx,
            )
            self._apply_step_scores(step_span, metadata)
            self._end_observation(step_span, end_time=None)
        except Exception as e:
            warnings.warn(f"Langfuse log_step failed (step_{step_index}): {e!s}")
            try:
                self.log_action_generation(
                    step_index=step_index,
                    compute_stage=stage,
                    prompt=prompt,
                    output=action_output,
                    model_name=model_name,
                    metadata=metadata,
                )
            except Exception:
                return

    def episode_end(
        self,
        *,
        output: dict[str, Any] | None = None,
        final_tags: list[str] | None = None,
    ) -> None:
        if self._otel_api and self._root_span is not None:
            try:
                if output:
                    try:
                        self._root_span.update(output=dict(output))
                    except Exception:
                        pass
                if self._episode_cm is not None:
                    try:
                        self._episode_cm.__exit__(None, None, None)
                    except Exception:
                        self._root_span.end()
                else:
                    self._root_span.end()
            except Exception as e:
                warnings.warn(f"Langfuse root span end failed: {e!s}")
            if self._propagate_cm is not None:
                try:
                    self._propagate_cm.__exit__(None, None, None)
                except Exception:
                    pass
            self._root_span = None
            self._episode_cm = None
            self._propagate_cm = None
            self._trace_id = None
            self._episode_tags = []
            self._session_id = None
            self._trace_name = None
            flush = getattr(self._client, "flush", None)
            if callable(flush):
                try:
                    flush()
                except Exception as e:
                    warnings.warn(f"Langfuse flush failed: {e!s}")
            return

        if self._root_span is not None:
            try:
                self._root_span.end()
            except Exception as e:
                warnings.warn(f"Langfuse root span end failed: {e!s}")
            self._root_span = None
        payload: dict[str, Any] = {}
        if output:
            payload["output"] = dict(output)
        tags = list(self._episode_tags)
        if final_tags:
            tags.extend([t for t in final_tags if t])
        if tags:
            payload["tags"] = tags
        if payload:
            upd_trace = getattr(self._client, "update_trace", None)
            if callable(upd_trace) and self._trace_id:
                try:
                    upd_trace(trace_id=self._trace_id, **payload)
                except TypeError:
                    try:
                        upd_trace(id=self._trace_id, **payload)
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                upd = getattr(self._client, "update_current_trace", None)
                if callable(upd):
                    try:
                        upd(**payload)
                    except Exception:
                        try:
                            upd(payload)
                        except Exception:
                            pass
        self._trace_id = None
        self._episode_tags = []
        self._session_id = None
        self._trace_name = None
        flush = getattr(self._client, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception as e:
                warnings.warn(f"Langfuse flush failed: {e!s}")


def _nonempty_str(val: Any) -> bool:
    return bool(val) and str(val).strip() != ""


def log_langfuse_startup_status(
    config: dict[str, Any] | None,
    *,
    dotenv_info: dict[str, Any] | None = None,
) -> None:
    cfg = (config or {}).get("tracing") or {}
    if not bool(cfg.get("langfuse_enabled")):
        return

    from src.utils.run_progress import log

    if dotenv_info:
        if dotenv_info.get("loaded"):
            dotenv_msg = f".env loaded=yes path={dotenv_info.get('env_path', '')}"
        else:
            reason = dotenv_info.get("reason") or "unknown"
            dotenv_msg = f".env loaded=no path={dotenv_info.get('env_path', '')} reason={reason}"
    else:
        dotenv_msg = ".env status=unknown (script did not pass dotenv_info)"

    env_pk = _nonempty_str(os.environ.get("LANGFUSE_PUBLIC_KEY"))
    env_sk = _nonempty_str(os.environ.get("LANGFUSE_SECRET_KEY"))
    env_host = _nonempty_str(os.environ.get("LANGFUSE_HOST"))
    yaml_pk = _nonempty_str(cfg.get("langfuse_public_key"))
    yaml_sk = _nonempty_str(cfg.get("langfuse_secret_key"))
    yaml_host = _nonempty_str(cfg.get("langfuse_host"))

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY") or cfg.get("langfuse_public_key")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY") or cfg.get("langfuse_secret_key")
    creds_ok = _nonempty_str(public_key) and _nonempty_str(secret_key)

    try:
        import langfuse  # noqa: F401

        sdk_ok = True
    except ImportError:
        sdk_ok = False

    will_trace = creds_ok and sdk_ok
    log(
        f"Langfuse startup — {dotenv_msg} | "
        f"env_vars_nonempty public={env_pk} secret={env_sk} host={env_host} | "
        f"yaml_keys_nonempty public={yaml_pk} secret={yaml_sk} host={yaml_host} | "
        f"credentials_resolved={'yes' if creds_ok else 'no'} | "
        f"sdk_installed={'yes' if sdk_ok else 'no'} | "
        f"tracing_active={'yes' if will_trace else 'no'}"
    )


def build_trace_hook(tracing_cfg: dict[str, Any] | None) -> TraceHook:
    cfg = tracing_cfg or {}
    if not bool(cfg.get("langfuse_enabled")):
        return NullTraceHook()

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY") or cfg.get("langfuse_public_key")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY") or cfg.get("langfuse_secret_key")
    host = os.environ.get("LANGFUSE_HOST") or cfg.get("langfuse_host")

    if not public_key or not secret_key:
        warnings.warn(
            "tracing.langfuse_enabled is true but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
            "(or YAML keys) are missing; Langfuse tracing disabled."
        )
        return NullTraceHook()

    try:
        from langfuse import Langfuse
    except ImportError:
        warnings.warn("tracing.langfuse_enabled but package 'langfuse' is not installed.")
        return NullTraceHook()

    kwargs: dict[str, Any] = {
        "public_key": str(public_key),
        "secret_key": str(secret_key),
    }
    if host:
        kwargs["host"] = str(host).strip()

    try:
        client = Langfuse(**kwargs)
    except Exception as e:
        warnings.warn(f"Langfuse client init failed: {e!s}")
        return NullTraceHook()

    return LangfuseTraceHook(client)


def optional_trace_hook_from_config(
    config: dict[str, Any] | None,
    *,
    dotenv_info: dict[str, Any] | None = None,
) -> TraceHook | None:
    cfg = (config or {}).get("tracing") or {}
    if not bool(cfg.get("langfuse_enabled")):
        return None
    log_langfuse_startup_status(config, dotenv_info=dotenv_info)
    return build_trace_hook(cfg)
