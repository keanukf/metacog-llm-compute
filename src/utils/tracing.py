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

    # Optional richer API (used when implemented):
    def start_step_observation(  # pragma: no cover - protocol only
        self,
        *,
        step_index: int,
        stage: str,
        observation: str,
        metadata: dict[str, Any] | None,
    ):
        """
        Context manager that brackets the *entire step wall time*.

        Duration in Langfuse is derived from observation start/end, so this must be entered
        before step work begins and exited after all step work completes.
        """
        ...

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
    ) -> None:
        """Create action + VC observations as children of an already-active step span."""
        ...

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


class LangfuseTraceHook:
    """
    One Langfuse trace per episode (shared trace_id), one generation observation per env step.
    Uses the Langfuse Python SDK v3+ API (``start_observation`` + ``TraceContext``).
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._trace_id: str | None = None
        self._root_span: Any = None
        self._episode_cm: Any = None
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
            # Langfuse SDKs differ: some accept tags directly on observations, others don't.
            self._supports_observation_tags = "tags" in sig.parameters
            self._supports_observation_times = ("start_time" in sig.parameters) or (
                "end_time" in sig.parameters
            )
        except Exception:
            self._supports_parent_observation_id = False
            self._supports_observation_tags = False
            self._supports_observation_times = False

        # OTel-based Python SDK (v3+): prefer context-manager APIs for correct nesting & timing.
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
    def _extract_dt(meta: dict[str, Any] | None, key: str) -> Any | None:
        if not meta:
            return None
        val = meta.get(key)
        if val is None:
            return None
        return val

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
    ) -> Any:
        """
        Best-effort attempt to create a generation as a child of ``parent_span``.
        Falls back to a flat observation when SDK lacks a parent-child API.
        """
        gen_kw: dict[str, Any] = {}
        if self._supports_observation_tags and tags:
            gen_kw["tags"] = list(tags)
        if self._supports_observation_times:
            if start_time is not None:
                gen_kw["start_time"] = start_time
            if end_time is not None:
                gen_kw["end_time"] = end_time

        # 1) Preferred: client.start_observation with parent_observation_id
        parent_id = getattr(parent_span, "id", None)
        if parent_id is not None:
            try:
                return self._client.start_observation(
                    name=name,
                    as_type="generation",
                    trace_context=ctx,
                    parent_observation_id=parent_id,
                    model=model,
                    input=input_text,
                    output=output_text,
                    metadata=metadata,
                    **gen_kw,
                )
            except TypeError:
                pass

        # 2) Some SDKs expose a helper on the parent observation
        for meth_name in ("generation", "start_generation", "start_observation"):
            meth = getattr(parent_span, meth_name, None)
            if not callable(meth):
                continue
            try:
                if meth_name == "start_observation":
                    return meth(
                        name=name,
                        as_type="generation",
                        trace_context=ctx,
                        model=model,
                        input=input_text,
                        output=output_text,
                        metadata=metadata,
                        **gen_kw,
                    )
                return meth(
                    name=name,
                    model=model,
                    input=input_text,
                    output=output_text,
                    metadata=metadata,
                    **gen_kw,
                )
            except TypeError:
                continue
            except Exception:
                break

        # 3) Flat fallback (still has correct timestamps)
        return self._client.start_observation(
            name=name,
            as_type="generation",
            trace_context=ctx,
            model=model,
            input=input_text,
            output=output_text,
            metadata=metadata,
            **gen_kw,
        )

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
        self._episode_tags = list(tags or [])
        self._session_id = session_id
        self._trace_name = trace_name
        meta = dict(metadata or {})
        meta["episode_id"] = episode_id
        try:
            meta["session_id"] = session_id
            meta["tags"] = list(self._episode_tags)
            if trace_name:
                meta["trace_name"] = trace_name

            if self._otel_api:
                # v3/OTel SDK: keep an *active* root observation open across the episode.
                # This ensures the trace root is the episode (not step_0) and names are correct.
                root_name = str(trace_name) if trace_name else "metacog_episode"
                self._episode_cm = self._client.start_as_current_observation(
                    as_type="span",
                    name=root_name,
                    metadata=meta,
                )
                self._root_span = self._episode_cm.__enter__()
                self._trace_id = getattr(self._root_span, "trace_id", None)

                # Best-effort trace fields (name/session/tags) via update_trace when available.
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

            # Legacy (pre-OTel) SDK path
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
                # CI / minimal installs: no langfuse package; client may still be injected (tests).
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
                else:
                    upd = getattr(self._client, "update_current_trace", None)
                    if callable(upd):
                        try:
                            upd(**trace_fields)
                        except Exception as e:
                            try:
                                upd(trace_fields)
                            except Exception:
                                warnings.warn(f"Langfuse update_current_trace failed: {e!s}")
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
            step_meta = dict(metadata or {})
            step_meta["stage"] = stage
            if self._session_id:
                step_meta["session_id"] = self._session_id
            if self._episode_tags:
                step_meta["trace_tags"] = list(self._episode_tags)
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
            # If stages provide subcalls, log them explicitly instead of one merged action generation.
            did_subcalls = False
            if stage == "C1" and isinstance(subcalls, list) and subcalls:
                for sc in subcalls:
                    if not isinstance(sc, dict):
                        continue
                    kind = str(sc.get("kind") or "").strip().lower()
                    if kind not in {"cot", "verify"}:
                        continue
                    sc_meta = dict(metadata or {})
                    sc_meta["stage"] = stage
                    sc_meta["kind"] = kind
                    sc_meta["lm_call_index"] = 1 if kind == "cot" else 2
                    sc_meta["counts_toward_compute"] = True
                    if isinstance(sc.get("tokens_generated"), int):
                        sc_meta["tokens_generated"] = int(sc.get("tokens_generated"))
                    if isinstance(sc.get("temperature"), (int, float)):
                        sc_meta["temperature"] = float(sc.get("temperature"))
                    if strategy:
                        sc_meta["strategy"] = strategy
                    if self._session_id:
                        sc_meta["session_id"] = self._session_id
                    if self._episode_tags:
                        sc_meta["trace_tags"] = list(self._episode_tags)
                    gen = step_span.start_observation(
                        name=f"{kind}_{step_index}_{stage}",
                        as_type="generation",
                        model=model_name or "unknown",
                        input=str(sc.get("prompt") or ""),
                        metadata=sc_meta,
                    )
                    try:
                        gen.update(output=str(sc.get("response") or ""))
                    except Exception:
                        pass
                    try:
                        gen.end()
                    except Exception:
                        pass
                did_subcalls = True

            if stage == "C2" and isinstance(subcalls, list) and subcalls:
                for sc in subcalls:
                    if not isinstance(sc, dict):
                        continue
                    kind = str(sc.get("kind") or "").strip().lower()
                    if kind != "sample":
                        continue
                    sc_meta = dict(metadata or {})
                    sc_meta["stage"] = stage
                    sc_meta["kind"] = "sample"
                    sc_meta["counts_toward_compute"] = True
                    if isinstance(sc.get("sample_index"), int):
                        sc_meta["sample_index"] = int(sc.get("sample_index"))
                    if isinstance(sc.get("is_winner"), bool):
                        sc_meta["is_winner"] = bool(sc.get("is_winner"))
                    if isinstance(sc.get("tokens_generated"), int):
                        sc_meta["tokens_generated"] = int(sc.get("tokens_generated"))
                    if sc.get("mean_logprob") is not None:
                        sc_meta["mean_logprob"] = sc.get("mean_logprob")
                    if sc.get("tle") is not None:
                        sc_meta["tle"] = sc.get("tle")
                    if strategy:
                        sc_meta["strategy"] = strategy
                    if self._session_id:
                        sc_meta["session_id"] = self._session_id
                    if self._episode_tags:
                        sc_meta["trace_tags"] = list(self._episode_tags)
                    si = sc_meta.get("sample_index", 0)
                    gen = step_span.start_observation(
                        name=f"sample_{si}_{step_index}_{stage}",
                        as_type="generation",
                        model=model_name or "unknown",
                        input=str(sc.get("prompt") or ""),
                        metadata=sc_meta,
                    )
                    try:
                        gen.update(output=str(sc.get("response") or ""))
                    except Exception:
                        pass
                    try:
                        gen.end()
                    except Exception:
                        pass
                did_subcalls = True

            if not did_subcalls:
                action_meta = dict(metadata or {})
                action_meta["stage"] = stage
                action_meta["action"] = action
                if strategy:
                    action_meta["strategy"] = strategy
                if self._session_id:
                    action_meta["session_id"] = self._session_id
                if self._episode_tags:
                    action_meta["trace_tags"] = list(self._episode_tags)

                # Keep action + VC as children of step (hierarchy stable).
                action_gen = step_span.start_observation(
                    name=f"action_{step_index}_{stage}",
                    as_type="generation",
                    model=model_name or "unknown",
                    input=prompt,
                    metadata=action_meta,
                )
                try:
                    action_gen.update(output=action_output)
                except Exception:
                    pass

            if vc_prompt and vc_output:
                vc_meta = dict(metadata or {})
                vc_meta["stage"] = stage
                vc_meta["kind"] = "vc_followup"
                if self._session_id:
                    vc_meta["session_id"] = self._session_id
                if self._episode_tags:
                    vc_meta["trace_tags"] = list(self._episode_tags)
                with step_span.start_as_current_observation(
                    as_type="generation",
                    name=f"vc_followup_{step_index}",
                    model=model_name or "unknown",
                    input=vc_prompt,
                    metadata=vc_meta,
                ) as vc_gen:
                    vc_gen.update(output=vc_output)

            if not did_subcalls:
                try:
                    action_gen.end()
                except Exception:
                    pass
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
            # OTel-based SDK path: use context managers to guarantee nesting and timing.
            if self._otel_api and self._root_span is not None:
                step_meta = dict(metadata or {})
                step_meta["action"] = action
                if strategy:
                    step_meta["strategy"] = strategy
                if self._session_id:
                    step_meta["session_id"] = self._session_id
                if self._episode_tags:
                    step_meta["trace_tags"] = list(self._episode_tags)

                # Create step span (child of active episode root).
                with self._client.start_as_current_observation(
                    as_type="span",
                    name=f"step_{step_index}",
                    input=observation,
                    metadata=step_meta,
                ) as step_span:
                    did_subcalls = False
                    if stage == "C1" and isinstance(subcalls, list) and subcalls:
                        for sc in subcalls:
                            if not isinstance(sc, dict):
                                continue
                            kind = str(sc.get("kind") or "").strip().lower()
                            if kind not in {"cot", "verify"}:
                                continue
                            sc_meta = dict(metadata or {})
                            sc_meta["stage"] = stage
                            sc_meta["kind"] = kind
                            sc_meta["lm_call_index"] = 1 if kind == "cot" else 2
                            sc_meta["counts_toward_compute"] = True
                            if isinstance(sc.get("tokens_generated"), int):
                                sc_meta["tokens_generated"] = int(sc.get("tokens_generated"))
                            if isinstance(sc.get("temperature"), (int, float)):
                                sc_meta["temperature"] = float(sc.get("temperature"))
                            if self._session_id:
                                sc_meta["session_id"] = self._session_id
                            if self._episode_tags:
                                sc_meta["trace_tags"] = list(self._episode_tags)
                            gen = step_span.start_observation(
                                name=f"{kind}_{step_index}_{stage}",
                                as_type="generation",
                                model=model_name or "unknown",
                                input=str(sc.get("prompt") or ""),
                                metadata=sc_meta,
                            )
                            try:
                                gen.update(output=str(sc.get("response") or ""))
                            except Exception:
                                pass
                            try:
                                gen.end()
                            except Exception:
                                pass
                        did_subcalls = True

                    if stage == "C2" and isinstance(subcalls, list) and subcalls:
                        for sc in subcalls:
                            if not isinstance(sc, dict):
                                continue
                            kind = str(sc.get("kind") or "").strip().lower()
                            if kind != "sample":
                                continue
                            sc_meta = dict(metadata or {})
                            sc_meta["stage"] = stage
                            sc_meta["kind"] = "sample"
                            sc_meta["counts_toward_compute"] = True
                            if isinstance(sc.get("sample_index"), int):
                                sc_meta["sample_index"] = int(sc.get("sample_index"))
                            if isinstance(sc.get("is_winner"), bool):
                                sc_meta["is_winner"] = bool(sc.get("is_winner"))
                            if isinstance(sc.get("tokens_generated"), int):
                                sc_meta["tokens_generated"] = int(sc.get("tokens_generated"))
                            if sc.get("mean_logprob") is not None:
                                sc_meta["mean_logprob"] = sc.get("mean_logprob")
                            if sc.get("tle") is not None:
                                sc_meta["tle"] = sc.get("tle")
                            if self._session_id:
                                sc_meta["session_id"] = self._session_id
                            if self._episode_tags:
                                sc_meta["trace_tags"] = list(self._episode_tags)
                            si = sc_meta.get("sample_index", 0)
                            gen = step_span.start_observation(
                                name=f"sample_{si}_{step_index}_{stage}",
                                as_type="generation",
                                model=model_name or "unknown",
                                input=str(sc.get("prompt") or ""),
                                metadata=sc_meta,
                            )
                            try:
                                gen.update(output=str(sc.get("response") or ""))
                            except Exception:
                                pass
                            try:
                                gen.end()
                            except Exception:
                                pass
                        did_subcalls = True

                    if not did_subcalls:
                        action_meta = dict(metadata or {})
                        action_meta["stage"] = stage
                        if self._session_id:
                            action_meta["session_id"] = self._session_id
                        if self._episode_tags:
                            action_meta["trace_tags"] = list(self._episode_tags)

                        # Manual observation so we can extend its duration to include VC follow-up
                        # (Langfuse UI otherwise shows near-0s).
                        action_gen = step_span.start_observation(
                            name=f"action_{step_index}_{stage}",
                            as_type="generation",
                            model=model_name or "unknown",
                            input=prompt,
                            metadata=action_meta,
                        )
                        try:
                            action_gen.update(output=action_output)
                        except Exception:
                            pass

                    # VC follow-up stays a sibling under step_span (hierarchy unchanged).
                    if vc_prompt and vc_output:
                        vc_meta = dict(metadata or {})
                        vc_meta["stage"] = stage
                        vc_meta["kind"] = "vc_followup"
                        if self._session_id:
                            vc_meta["session_id"] = self._session_id
                        if self._episode_tags:
                            vc_meta["trace_tags"] = list(self._episode_tags)
                        with step_span.start_as_current_observation(
                            as_type="generation",
                            name=f"vc_followup_{step_index}",
                            model=model_name or "unknown",
                            input=vc_prompt,
                            metadata=vc_meta,
                        ) as vc_gen:
                            vc_gen.update(output=vc_output)

                    if not did_subcalls:
                        # End action generation last to give it a non-zero duration.
                        try:
                            action_gen.end()
                        except Exception:
                            pass

                return

            from langfuse.types import TraceContext

            ctx = TraceContext(trace_id=self._trace_id)
            step_meta = dict(metadata or {})
            step_meta["action"] = action
            if strategy:
                step_meta["strategy"] = strategy
            if self._session_id:
                step_meta["session_id"] = self._session_id
            step_kw: dict[str, Any] = {}
            if self._supports_observation_tags and self._episode_tags:
                step_kw["tags"] = list(self._episode_tags)
            elif self._episode_tags:
                # Back-compat: keep tags visible somewhere even when SDK doesn't support observation tags.
                step_meta["trace_tags"] = list(self._episode_tags)
            step_span = self._client.start_observation(
                name=f"step_{step_index}",
                as_type="span",
                trace_context=ctx,
                input=observation,
                metadata=step_meta,
                **step_kw,
            )
            did_subcalls = False
            if stage == "C1" and isinstance(subcalls, list) and subcalls:
                for sc in subcalls:
                    if not isinstance(sc, dict):
                        continue
                    kind = str(sc.get("kind") or "").strip().lower()
                    if kind not in {"cot", "verify"}:
                        continue
                    sc_meta = dict(metadata or {})
                    sc_meta["stage"] = stage
                    sc_meta["kind"] = kind
                    sc_meta["lm_call_index"] = 1 if kind == "cot" else 2
                    sc_meta["counts_toward_compute"] = True
                    if isinstance(sc.get("tokens_generated"), int):
                        sc_meta["tokens_generated"] = int(sc.get("tokens_generated"))
                    if isinstance(sc.get("temperature"), (int, float)):
                        sc_meta["temperature"] = float(sc.get("temperature"))
                    if self._session_id:
                        sc_meta["session_id"] = self._session_id
                    if (not self._supports_observation_tags) and self._episode_tags:
                        sc_meta["trace_tags"] = list(self._episode_tags)
                    gen = self._start_generation_child(
                        parent_span=step_span,
                        ctx=ctx,
                        name=f"{kind}_{step_index}_{stage}",
                        model=model_name or "unknown",
                        input_text=str(sc.get("prompt") or ""),
                        output_text=str(sc.get("response") or ""),
                        metadata=sc_meta,
                        tags=list(self._episode_tags) if self._episode_tags else None,
                        start_time=None,
                        end_time=None,
                    )
                    self._end_observation(gen, end_time=None)
                did_subcalls = True

            if stage == "C2" and isinstance(subcalls, list) and subcalls:
                for sc in subcalls:
                    if not isinstance(sc, dict):
                        continue
                    kind = str(sc.get("kind") or "").strip().lower()
                    if kind != "sample":
                        continue
                    sc_meta = dict(metadata or {})
                    sc_meta["stage"] = stage
                    sc_meta["kind"] = "sample"
                    sc_meta["counts_toward_compute"] = True
                    if isinstance(sc.get("sample_index"), int):
                        sc_meta["sample_index"] = int(sc.get("sample_index"))
                    if isinstance(sc.get("is_winner"), bool):
                        sc_meta["is_winner"] = bool(sc.get("is_winner"))
                    if isinstance(sc.get("tokens_generated"), int):
                        sc_meta["tokens_generated"] = int(sc.get("tokens_generated"))
                    if sc.get("mean_logprob") is not None:
                        sc_meta["mean_logprob"] = sc.get("mean_logprob")
                    if sc.get("tle") is not None:
                        sc_meta["tle"] = sc.get("tle")
                    if self._session_id:
                        sc_meta["session_id"] = self._session_id
                    if (not self._supports_observation_tags) and self._episode_tags:
                        sc_meta["trace_tags"] = list(self._episode_tags)
                    si = sc_meta.get("sample_index", 0)
                    gen = self._start_generation_child(
                        parent_span=step_span,
                        ctx=ctx,
                        name=f"sample_{si}_{step_index}_{stage}",
                        model=model_name or "unknown",
                        input_text=str(sc.get("prompt") or ""),
                        output_text=str(sc.get("response") or ""),
                        metadata=sc_meta,
                        tags=list(self._episode_tags) if self._episode_tags else None,
                        start_time=None,
                        end_time=None,
                    )
                    self._end_observation(gen, end_time=None)
                did_subcalls = True

            if not did_subcalls:
                # Propagate caller metadata (e.g. TLE/VC/correctness) to the action generation
                # so it is visible where people typically inspect model outputs in Langfuse.
                action_meta = dict(metadata or {})
                action_meta["stage"] = stage
                if self._session_id:
                    action_meta["session_id"] = self._session_id
                if (not self._supports_observation_tags) and self._episode_tags:
                    action_meta["trace_tags"] = list(self._episode_tags)
                action_gen = self._start_generation_child(
                    parent_span=step_span,
                    ctx=ctx,
                    name=f"action_{step_index}_{stage}",
                    model=model_name or "unknown",
                    input_text=prompt,
                    output_text=action_output,
                    metadata=action_meta,
                    tags=list(self._episode_tags) if self._episode_tags else None,
                    start_time=None,
                    end_time=None,
                )
                self._end_observation(action_gen, end_time=None)
            if vc_prompt and vc_output:
                vc_meta = dict(metadata or {})
                vc_meta["stage"] = stage
                vc_meta["kind"] = "vc_followup"
                if self._session_id:
                    vc_meta["session_id"] = self._session_id
                if (not self._supports_observation_tags) and self._episode_tags:
                    vc_meta["trace_tags"] = list(self._episode_tags)
                vc_gen = self._start_generation_child(
                    parent_span=step_span,
                    ctx=ctx,
                    name=f"vc_followup_{step_index}",
                    model=model_name or "unknown",
                    input_text=vc_prompt,
                    output_text=vc_output,
                    metadata=vc_meta,
                    tags=list(self._episode_tags) if self._episode_tags else None,
                    start_time=None,
                    end_time=None,
                )
                self._end_observation(vc_gen, end_time=None)
            self._end_observation(step_span, end_time=None)
        except Exception as e:
            warnings.warn(f"Langfuse log_step failed (step_{step_index}): {e!s}")
            # Fallback to the flat action-generation log
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
        # OTel SDK: update root span output and end it.
        if self._otel_api and self._root_span is not None:
            try:
                if output:
                    try:
                        self._root_span.update(output=dict(output))
                    except Exception:
                        pass
                # Close the active root observation context if we opened one.
                if self._episode_cm is not None:
                    try:
                        self._episode_cm.__exit__(None, None, None)
                    except Exception:
                        # Fall back to ending the span object
                        self._root_span.end()
                else:
                    self._root_span.end()
            except Exception as e:
                warnings.warn(f"Langfuse root span end failed: {e!s}")
            self._root_span = None
            self._episode_cm = None
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
            # Prefer explicit trace-id update when available.
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
    """
    One timestamped line when ``tracing.langfuse_enabled`` is true: .env load outcome,
    whether Langfuse keys resolve from env/YAML (no secret values), SDK presence, and whether
    tracing will actually run.
    """
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
    """
    Return a Langfuse-backed hook if ``langfuse_enabled`` and keys are available, else NullTraceHook.
    """
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
    """
    If ``tracing.langfuse_enabled`` is false, return None (skip hook construction).
    If true, log startup diagnostics (when ``dotenv_info`` is passed, includes .env load status),
    then return ``build_trace_hook`` result (possibly ``NullTraceHook`` if keys/SDK missing).
    """
    cfg = (config or {}).get("tracing") or {}
    if not bool(cfg.get("langfuse_enabled")):
        return None
    log_langfuse_startup_status(config, dotenv_info=dotenv_info)
    return build_trace_hook(cfg)
