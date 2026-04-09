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


class TraceHook(Protocol):
    """Per-episode tracing (file logging is handled separately in base_agent)."""

    def episode_start(self, episode_id: str, metadata: dict[str, Any] | None = None) -> None: ...

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

    def episode_end(self) -> None: ...


class NullTraceHook:
    """No-op tracer when Langfuse is off or unavailable."""

    def episode_start(self, episode_id: str, metadata: dict[str, Any] | None = None) -> None:
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

    def episode_end(self) -> None:
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

    def episode_start(self, episode_id: str, metadata: dict[str, Any] | None = None) -> None:
        self._trace_id = None
        self._root_span = None
        meta = dict(metadata or {})
        meta["episode_id"] = episode_id
        try:
            from langfuse.types import TraceContext

            self._trace_id = self._client.create_trace_id()
            ctx = TraceContext(trace_id=self._trace_id)
            self._root_span = self._client.start_observation(
                name="metacog_episode",
                as_type="span",
                trace_context=ctx,
                metadata=meta,
            )
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

    def episode_end(self) -> None:
        if self._root_span is not None:
            try:
                self._root_span.end()
            except Exception as e:
                warnings.warn(f"Langfuse root span end failed: {e!s}")
            self._root_span = None
        self._trace_id = None
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
