from __future__ import annotations

from typing import Any


def log_step_with_fallback(
    *,
    trace_hook: Any,
    step_index: int,
    stage: str,
    observation: str,
    action: str,
    prompt_full: str | None,
    response_full: str | None,
    vc_det: dict[str, Any] | None,
    trace_model_name: str | None,
    metadata: dict[str, Any],
    subcalls: list[dict[str, Any]] | None = None,
    strategy: str | None = None,
) -> None:
    """Attempt rich `log_step`, then fallback to `log_action_generation` if unavailable/failing."""
    if hasattr(trace_hook, "log_step"):
        try:
            kwargs: dict[str, Any] = {
                "step_index": step_index,
                "stage": stage,
                "observation": observation,
                "action": action,
                "prompt": prompt_full or "",
                "action_output": response_full or "",
                "vc_prompt": (vc_det or {}).get("vc_prompt") if vc_det else None,
                "vc_output": (vc_det or {}).get("vc_raw_text") if vc_det else None,
                "model_name": trace_model_name,
                "metadata": metadata,
                "subcalls": subcalls,
            }
            if strategy is not None:
                kwargs["strategy"] = strategy
            trace_hook.log_step(**kwargs)
            return
        except Exception:
            pass
    trace_hook.log_action_generation(
        step_index=step_index,
        compute_stage=stage,
        prompt=prompt_full or "",
        output=response_full or "",
        model_name=trace_model_name,
        metadata=metadata,
    )
