"""
Optional observability hooks: Langfuse cloud traces (no-op when disabled or missing SDK).

Configure via ``tracing`` YAML block and/or environment variables:
``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, optional ``LANGFUSE_HOST``.

Requires ``langfuse`` (see optional dependency group ``tracing`` in pyproject.toml).
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Protocol
import inspect


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
        self._episode_tags: list[str] = []
        self._session_id: str | None = None
        self._trace_name: str | None = None
        self._supports_parent_observation_id = False
        try:
            sig = inspect.signature(getattr(self._client, "start_observation"))
            self._supports_parent_observation_id = "parent_observation_id" in sig.parameters
        except Exception:
            self._supports_parent_observation_id = False

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
        self._episode_tags = list(tags or [])
        self._session_id = session_id
        self._trace_name = trace_name
        meta = dict(metadata or {})
        meta["episode_id"] = episode_id
        try:
            from langfuse.types import TraceContext

            self._trace_id = self._client.create_trace_id()
            ctx = TraceContext(trace_id=self._trace_id)
            meta["session_id"] = session_id
            meta["tags"] = list(self._episode_tags)
            if trace_name:
                meta["trace_name"] = trace_name
            self._root_span = self._client.start_observation(
                name="metacog_episode",
                as_type="span",
                trace_context=ctx,
                metadata=meta,
            )
            upd = getattr(self._client, "update_current_trace", None)
            if callable(upd):
                payload: dict[str, Any] = {}
                if trace_name:
                    payload["name"] = str(trace_name)
                if session_id:
                    payload["session_id"] = str(session_id)
                if self._episode_tags:
                    payload["tags"] = list(self._episode_tags)
                if payload:
                    try:
                        upd(**payload)
                    except Exception as e:
                        # Some SDK versions use positional dict payload instead of kwargs.
                        try:
                            upd(payload)
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
            gen = self._client.start_observation(
                name=f"step_{step_index}_{compute_stage}",
                as_type="generation",
                trace_context=ctx,
                model=model_name or "unknown",
                input=prompt,
                output=output,
                metadata=metadata or {},
            )
            gen.end()
        except Exception as e:
            warnings.warn(
                f"Langfuse log_action_generation failed (step_{step_index}_{compute_stage}): {e!s}"
            )

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
    ) -> None:
        if self._trace_id is None:
            return
        try:
            from langfuse.types import TraceContext

            ctx = TraceContext(trace_id=self._trace_id)
            step_meta = dict(metadata or {})
            step_meta["action"] = action
            if strategy:
                step_meta["strategy"] = strategy
            if self._session_id:
                step_meta["session_id"] = self._session_id
            if self._episode_tags:
                step_meta["trace_tags"] = list(self._episode_tags)
            step_span = self._client.start_observation(
                name=f"step_{step_index}",
                as_type="span",
                trace_context=ctx,
                input=observation,
                metadata=step_meta,
            )
            gen_kw: dict[str, Any] = {}
            if self._supports_parent_observation_id:
                parent_id = getattr(step_span, "id", None)
                if parent_id:
                    gen_kw["parent_observation_id"] = parent_id
            action_gen = self._client.start_observation(
                name=f"action_{step_index}_{stage}",
                as_type="generation",
                trace_context=ctx,
                model=model_name or "unknown",
                input=prompt,
                output=action_output,
                metadata={
                    "stage": stage,
                    "session_id": self._session_id,
                    "trace_tags": list(self._episode_tags),
                },
                **gen_kw,
            )
            action_gen.end()
            if vc_prompt and vc_output:
                vc_gen = self._client.start_observation(
                    name=f"vc_followup_{step_index}",
                    as_type="generation",
                    trace_context=ctx,
                    model=model_name or "unknown",
                    input=vc_prompt,
                    output=vc_output,
                    metadata={
                        "stage": stage,
                        "kind": "vc_followup",
                        "session_id": self._session_id,
                        "trace_tags": list(self._episode_tags),
                    },
                    **gen_kw,
                )
                vc_gen.end()
            step_span.end()
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
        if self._root_span is not None:
            try:
                self._root_span.end()
            except Exception as e:
                warnings.warn(f"Langfuse root span end failed: {e!s}")
            self._root_span = None
        upd = getattr(self._client, "update_current_trace", None)
        if callable(upd):
            payload: dict[str, Any] = {}
            if output:
                payload["output"] = dict(output)
            tags = list(self._episode_tags)
            if final_tags:
                tags.extend([t for t in final_tags if t])
            if tags:
                payload["tags"] = tags
            if payload:
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
