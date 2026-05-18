"""
Minimal agent loop: observation -> LM call -> action -> next observation.
No framework (no LangChain/LlamaIndex). Each compute stage is a clear function.
"""

from __future__ import annotations

import contextlib
import random
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.agent.allocator import allocate
from src.agent.compute_stages import get_step_fn
from src.agent.history_utils import compact_history_for_prompt, truncate_for_history
from src.agent.step_results import normalize_step_result
from src.agent.trace_integration import log_step_with_fallback
from src.utils.logging_utils import write_step_trace_line

# Backward compatibility for existing tests/imports during modularization.
_normalize_step_result = normalize_step_result
_truncate_for_history = truncate_for_history
_compact_history_for_prompt = compact_history_for_prompt

# Compute stage step: (observation, history, model) -> (action, tle_or_none, vc_or_none, tokens_used, lm_calls_this_step)
# Backward compat: step may return 3- or 4-tuple; defaults fill in missing values.
StepFn = Callable[
    [str, list[str], Any],
    tuple[str, dict | None, float | None]
    | tuple[str, dict | None, float | None, int]
    | tuple[str, dict | None, float | None, int, int],
]

_RECIPE_BLOCK_RE = re.compile(r"(Recipe\s*#\d+[\s\S]*?)\n\s*\n", re.IGNORECASE)


def _extract_recipe_block(text: str) -> str | None:
    if not text:
        return None
    m = _RECIPE_BLOCK_RE.search(text)
    if m:
        return (m.group(1) or "").strip()
    # Fallback: keep from "Recipe #" to end if no blank-line terminator exists.
    idx = text.lower().find("recipe #")
    if idx >= 0:
        return text[idx:].strip()
    return None


def _copy_step_results(env: Any) -> list[dict[str, Any]] | None:
    """Shallow copy env.step_results so callers cannot mutate episode records via the return dict."""
    raw = getattr(env, "step_results", None)
    if raw is None:
        return None
    return [dict(d) for d in raw]


def _signal_for_next_step(
    tle: dict[str, float] | None,
    vc: float | None,
) -> dict[str, Any] | None:
    """Build allocator signal from the last step's TLE / VC for the *next* step."""
    sig: dict[str, Any] = {}
    if tle is not None and "mean_entropy" in tle:
        sig["mean_entropy"] = tle["mean_entropy"]
    if vc is not None:
        sig["vc"] = vc
    return sig if sig else None


def run_episode(
    env: Any,
    model: Any,
    compute_stage: str,
    step_fn: StepFn | None = None,
    max_steps: int = 20,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    *,
    save_logprob_distributions: bool = False,
    save_vc_distributions: bool = False,
    save_step_traces: bool = False,
    episode_id: str | None = None,
    trace_output_dir: str | Path | None = None,
    trace_model_name: str | None = None,
    trace_hook: Any | None = None,
    history_keep_last_pairs: int | None = None,
    history_max_obs_chars: int = 0,
    history_current_obs_max_chars: int = 0,
    history_obs_head_ratio: float = 0.15,
    pin_recipe: bool = False,
    trace_session_id: str | None = None,
    trace_tags: list[str] | None = None,
    trace_name: str | None = None,
) -> dict[str, Any]:
    """
    Run one episode: reset env, then loop observation -> step_fn -> env.step until done.

    Args:
        env: Environment with reset() and step(action); has .observation and .done.
        model: Model wrapper with generate(prompt, logprobs=...).
        compute_stage: "C0" | "C1" | "C2" (used to select step_fn if not provided).
        step_fn: If None, a default stub step is used (returns dummy action).
        max_steps: Cap on steps per episode.
        on_step: Optional callback after each env step; dict keys include step_index,
            episode_steps, max_steps, env_done, compute_stage, lm_calls_this_step, total_lm_calls.
        save_logprob_distributions: If True, ``step_fn`` should return 7- or 9-tuples with raw action
            logprob lists; result includes ``logprob_raw_per_step`` for sidecar export.
        save_vc_distributions: If True, include full VC follow-up records in ``vc_detail_per_step``
            (and sidecar JSON when the runner writes it).
        save_step_traces: If True, append one JSON line per env step to
            ``trace_{episode_id}.jsonl`` under ``trace_output_dir`` (full prompt/response).
        episode_id: Used for trace filename when ``save_step_traces`` is True.
        trace_output_dir: Directory for JSONL trace files.
        trace_model_name: Optional model label for Langfuse metadata.
        trace_hook: Optional observability hook (e.g. Langfuse) with ``episode_start`` /
            ``log_action_generation`` / ``episode_end``.

    Returns:
        Dict with keys: steps, task_success, lm_calls, tokens, wall_clock_time,
        tle_per_step (optional), vc_per_step (optional),
        step_correctness (optional): shallow copy of env.step_results when present.
        logprob_raw_per_step (optional): list of per-token logprob dicts per env step when enabled.
        vc_detail_per_step (optional): rich VC metadata per step when follow-up VC is used.
    """
    obs = env.reset()
    # Keep reset() output in history: many envs (e.g. TextWorld) do not repeat the full scene
    # on every step, only the latest feedback + state. Without this, step ≥1 prompts lose the
    # opening game text.
    history: list[str] = [
        f"OBSERVATION: {_truncate_for_history(obs, max_chars=history_max_obs_chars, head_ratio=history_obs_head_ratio)}"
    ]
    pinned_recipe: str | None = None
    steps = 0
    total_lm_calls = 0
    total_tokens_generated = 0
    tle_per_step: list[dict | None] = []
    vc_per_step: list[float | None] = []
    logprob_raw_per_step: list[list[dict[str, Any]] | None] = []
    vc_detail_per_step: list[dict[str, Any] | None] = []
    steps_detail: list[dict[str, Any]] = []
    if step_fn is None:

        def _stub_step(
            o: str, h: list[str], m: Any
        ) -> tuple[str, dict | None, float | None, int, int]:
            return "go north", None, None, 0, 0

        step_fn = _stub_step

    trace_path: Path | None = None
    if save_step_traces:
        if not episode_id:
            warnings.warn(
                "save_step_traces is True but episode_id is missing; step traces not written."
            )
        elif not trace_output_dir:
            warnings.warn(
                "save_step_traces is True but trace_output_dir is missing; step traces not written."
            )
        else:
            trace_path = Path(trace_output_dir) / f"trace_{episode_id}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("", encoding="utf-8")

    ep_id_for_trace = episode_id or "ep_unknown"
    if trace_hook is not None:
        meta = {"compute_stage": compute_stage, "model": trace_model_name or ""}
        try:
            trace_hook.episode_start(
                ep_id_for_trace,
                metadata=meta,
                session_id=trace_session_id,
                tags=trace_tags,
                trace_name=trace_name,
            )
        except TypeError:
            # Backward compatible with older hook signature.
            trace_hook.episode_start(ep_id_for_trace, metadata=meta)

    t_start = time.perf_counter()
    try:
        while not getattr(env, "done", False) and steps < max_steps:
            step_obs = obs
            step_obs_for_prompt = _truncate_for_history(
                step_obs,
                max_chars=history_current_obs_max_chars,
                head_ratio=history_obs_head_ratio,
            )
            history_snapshot = list(history)
            step_start_dt = datetime.now(timezone.utc).isoformat()

            step_cm = contextlib.nullcontext(None)
            if trace_hook is not None and hasattr(trace_hook, "start_step_observation"):
                try:
                    step_cm = trace_hook.start_step_observation(
                        step_index=steps,
                        stage=compute_stage,
                        observation=step_obs,
                        metadata={"compute_stage": compute_stage, "model": trace_model_name or ""},
                    )
                except Exception:
                    step_cm = contextlib.nullcontext(None)

            with step_cm as step_span:
                t0 = time.perf_counter()
                history_for_prompt = _compact_history_for_prompt(
                    history, keep_last_pairs=history_keep_last_pairs
                )
                if pin_recipe and pinned_recipe:
                    history_for_prompt = [
                        history_for_prompt[0],
                        f"PINNED RECIPE:\n{pinned_recipe}",
                        *history_for_prompt[1:],
                    ]
                raw = step_fn(step_obs_for_prompt, history_for_prompt, model)
                step_wall_time_s = time.perf_counter() - t0
                action, tle, vc, tokens_used, lm_calls_this_step, log_raw, vc_det, prompt_full, response_full, call_detail = (
                    normalize_step_result(raw)
                )

                tle_per_step.append(tle)
                vc_per_step.append(vc)
                if save_logprob_distributions:
                    logprob_raw_per_step.append(log_raw)
                vc_detail_per_step.append(vc_det)
                total_tokens_generated += tokens_used
                total_lm_calls += lm_calls_this_step

                # Emit Langfuse children while the step span is active (keeps hierarchy stable).
                lf_meta_base: dict[str, Any] = {
                    "tle": tle,
                    "vc": vc,
                    "tokens_generated": int(tokens_used),
                    "lm_calls": int(lm_calls_this_step),
                    "step_wall_time_s": float(step_wall_time_s),
                    "step_start_time_utc": step_start_dt,
                }
                if (
                    trace_hook is not None
                    and step_span is not None
                    and hasattr(trace_hook, "log_step_children")
                ):
                    try:
                        subcalls = None
                        if isinstance(call_detail, dict) and isinstance(
                            call_detail.get("subcalls"), list
                        ):
                            subcalls = call_detail.get("subcalls")
                        trace_hook.log_step_children(
                            step_span=step_span,
                            step_index=steps,
                            stage=compute_stage,
                            action=action,
                            prompt=prompt_full or "",
                            action_output=response_full or "",
                            vc_prompt=(vc_det or {}).get("vc_prompt") if vc_det else None,
                            vc_output=(vc_det or {}).get("vc_raw_text") if vc_det else None,
                            model_name=trace_model_name,
                            metadata=lf_meta_base,
                            subcalls=subcalls,
                        )
                    except Exception:
                        pass

                row: dict[str, Any] = {
                    "step_index": steps,
                    "compute_stage": compute_stage,
                    "action": action,
                    "tokens_generated": int(tokens_used),
                    "lm_calls_this_step": int(lm_calls_this_step),
                    "step_wall_time_s": float(step_wall_time_s),
                    "tle": tle,
                    "vc": vc,
                    "correctness": None,
                    "observation_length_chars": len(step_obs or ""),
                }
                if compute_stage == "C2" and isinstance(call_detail, dict):
                    for k in ("vote_agreement", "unique_actions", "winner_index", "tie_broken"):
                        if k in call_detail:
                            row[k] = call_detail.get(k)
                if vc_det:
                    row["vc_prompt"] = vc_det.get("vc_prompt")
                    row["vc_raw_text"] = vc_det.get("vc_raw_text")
                    row["vc_pattern_matched"] = vc_det.get("vc_pattern_matched")
                    row["vc_tokens_used"] = vc_det.get("vc_tokens_used")
                    if save_vc_distributions and vc_det.get("vc_logprobs") is not None:
                        row["vc_logprobs"] = vc_det.get("vc_logprobs")
                steps_detail.append(row)

                # Environment transition is part of the full-step duration (per user choice).
                obs = env.step(action)
            if pin_recipe and pinned_recipe is None:
                rb = _extract_recipe_block(obs)
                if rb:
                    pinned_recipe = rb
            correctness_after: str | None = None
            sr = getattr(env, "step_results", None)
            if isinstance(sr, list) and sr:
                correctness_after = sr[-1].get("correctness")  # type: ignore[union-attr]
            if trace_path is not None:
                trace_rec: dict[str, Any] = {
                    "step_index": steps,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "compute_stage": compute_stage,
                    "prompt_full": prompt_full or "",
                    "response_full": response_full or "",
                    "call_detail": call_detail or None,
                    "action_parsed": action,
                    "observation_before": step_obs,
                    "observation_after": obs,
                    "history_snapshot": history_snapshot,
                    "tle": tle,
                    "vc": vc,
                    "correctness": correctness_after,
                    "step_wall_time_s": float(step_wall_time_s),
                    "lm_calls": int(lm_calls_this_step),
                    "tokens_generated": int(tokens_used),
                }
                if isinstance(call_detail, dict):
                    subcalls = call_detail.get("subcalls")
                    if isinstance(subcalls, list):
                        for sc in subcalls:
                            if not isinstance(sc, dict):
                                continue
                            kind = str(sc.get("kind") or "").strip().lower()
                            if kind == "cot":
                                trace_rec["cot_prompt"] = sc.get("prompt") or ""
                                trace_rec["cot_response"] = sc.get("response") or ""
                            elif kind == "verify":
                                trace_rec["verify_prompt"] = sc.get("prompt") or ""
                                trace_rec["verify_response"] = sc.get("response") or ""
                if vc_det:
                    trace_rec["vc_followup_prompt"] = vc_det.get("vc_prompt")
                    trace_rec["vc_followup_response"] = vc_det.get("vc_raw_text")
                write_step_trace_line(trace_path, trace_rec)
            if trace_hook is not None:
                lf_meta: dict[str, Any] = {
                    "tle": tle,
                    "vc": vc,
                    "correctness": correctness_after,
                    "tokens_generated": int(tokens_used),
                    "lm_calls": int(lm_calls_this_step),
                    "step_wall_time_s": float(step_wall_time_s),
                }
                # If we did not create a step-span context, fall back to legacy step logging.
                if step_span is None:
                    subcalls = None
                    if isinstance(call_detail, dict) and isinstance(call_detail.get("subcalls"), list):
                        subcalls = call_detail.get("subcalls")
                    log_step_with_fallback(
                        trace_hook=trace_hook,
                        step_index=steps,
                        stage=compute_stage,
                        observation=step_obs,
                        action=action,
                        prompt_full=prompt_full,
                        response_full=response_full,
                        vc_det=vc_det,
                        trace_model_name=trace_model_name,
                        metadata=lf_meta,
                        subcalls=subcalls,
                    )
            # Maintain growing context as ACTION/OBSERVATION pairs.
            # This allows later prompts/traces to reconstruct the full interaction history.
            history.append(f"ACTION: {action}")
            history.append(
                f"OBSERVATION: {_truncate_for_history(obs, max_chars=history_max_obs_chars, head_ratio=history_obs_head_ratio)}"
            )
            steps += 1
            if on_step is not None:
                on_step(
                    {
                        "step_index": steps - 1,
                        "episode_steps": steps,
                        "max_steps": max_steps,
                        "env_done": bool(getattr(env, "done", False)),
                        "compute_stage": compute_stage,
                        "lm_calls_this_step": int(lm_calls_this_step),
                        "total_lm_calls": int(total_lm_calls),
                    }
                )
    finally:
        if trace_hook is not None:
            try:
                succ = bool(getattr(env, "task_success", False)) or bool(
                    getattr(env, "done", False)
                )
            except Exception:
                succ = False
            final_tags: list[str] = []
            if succ:
                final_tags.append("succeeded")
            else:
                final_tags.append("failed")
            if steps >= max_steps:
                final_tags.append("hit_step_cap")
            try:
                trace_hook.episode_end(
                    output={
                        "task_success": succ,
                        "steps": int(steps),
                        "total_lm_calls": int(total_lm_calls),
                        "total_tokens_generated": int(total_tokens_generated),
                    },
                    final_tags=final_tags,
                )
            except TypeError:
                trace_hook.episode_end()

    wall_clock_time = time.perf_counter() - t_start
    if hasattr(env, "task_success"):
        task_success = bool(getattr(env, "task_success"))
    else:
        task_success = bool(getattr(env, "done", False))
    step_correctness = _copy_step_results(env)
    if step_correctness is not None:
        by_idx: dict[int, dict[str, Any]] = {}
        for d in step_correctness:
            try:
                idx = int(d.get("step_index"))
            except Exception:
                continue
            by_idx[idx] = d
        for sd in steps_detail:
            idx = int(sd["step_index"])
            corr = by_idx.get(idx, {}).get("correctness")
            sd["correctness"] = corr if corr is not None else None
    normalized_compute_cost = (
        (total_lm_calls / (max_steps * 3)) if max_steps and (max_steps * 3) > 0 else 0.0
    )
    efficiency_score = (
        (float(task_success) / normalized_compute_cost) if normalized_compute_cost > 0 else None
    )
    out: dict[str, Any] = {
        "steps": steps,  # legacy
        "episode_length_steps": steps,
        "task_success": task_success,
        "lm_calls": steps,  # legacy (was step-count historically)
        "total_lm_calls": int(total_lm_calls),
        "tokens": int(total_tokens_generated),  # legacy
        "total_tokens_generated": int(total_tokens_generated),
        "normalized_compute_cost": float(normalized_compute_cost),
        "efficiency_score": efficiency_score,
        "wall_clock_time": wall_clock_time,
        "tle_per_step": tle_per_step,
        "vc_per_step": vc_per_step,
        "steps_detail": steps_detail,
        "step_correctness": step_correctness,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if save_logprob_distributions:
        out["logprob_raw_per_step"] = logprob_raw_per_step
    if any(x is not None for x in vc_detail_per_step):
        out["vc_detail_per_step"] = vc_detail_per_step
    return out


def run_adaptive_episode(
    env: Any,
    model: Any,
    strategy: str,
    *,
    max_steps: int = 20,
    rng: random.Random | None = None,
    allocate_fn: Callable[..., str] | None = None,
    step_fn_for_stage: Callable[[str], StepFn] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    save_logprob_distributions: bool = False,
    save_vc_distributions: bool = False,
    save_step_traces: bool = False,
    episode_id: str | None = None,
    trace_output_dir: str | Path | None = None,
    trace_model_name: str | None = None,
    trace_hook: Any | None = None,
    vc_mode: str = "inline",
    prompt_prefix: str = "",
    action_max_tokens: int | None = None,
    action_temperature: float | None = None,
    followup_max_tokens: int = 4,
    followup_temperature: float = 0.0,
    action_stop: list[str] | None = None,
    followup_max_context_chars: int | None = None,
    followup_cot_max_chars: int = 12000,
    vc_raw_completion_max_chars: int = 8000,
    vc_followup_instruction: str | None = None,
    c1_cot_temperature: float | None = None,
    c1_cot_max_tokens: int | None = None,
    c1_verify_temperature: float = 0.0,
    c1_verify_max_tokens: int | None = None,
    c1_verify_stop: list[str] | None = None,
    c1_verify_instruction: str | None = None,
    c2_n_samples: int = 3,
    c2_tie_break_seed: str | int | None = None,
    history_keep_last_pairs: int | None = None,
    history_max_obs_chars: int = 0,
    history_current_obs_max_chars: int = 0,
    history_obs_head_ratio: float = 0.15,
    pin_recipe: bool = False,
    trace_session_id: str | None = None,
    trace_tags: list[str] | None = None,
    trace_name: str | None = None,
) -> dict[str, Any]:
    """
    Run an episode where each step's compute stage comes from ``allocate_fn`` (default: allocator.allocate).

    The signal passed into allocate is built from the *previous* step's TLE / VC; the first step uses
    ``signal=None`` so adaptive strategies default to C0 until a signal exists.
    on_step: Optional callback after each env step (same keys as ``run_episode``, plus ``strategy``).

    Returns:
        Same keys as ``run_episode`` plus ``stage_per_step``: list of ``"C0"`` | ``"C1"`` | ``"C2"`` per step.
    """
    rng = rng or random.Random()
    alloc = allocate_fn or allocate
    if step_fn_for_stage is not None:
        resolve = step_fn_for_stage
    else:

        def resolve(stage: str) -> StepFn:
            return get_step_fn(
                stage,
                save_logprob_distributions=save_logprob_distributions,
                save_vc_distributions=save_vc_distributions,
                c2_n_samples=int(c2_n_samples),
                c2_tie_break_seed=c2_tie_break_seed,
                vc_mode=vc_mode,
                prompt_prefix=prompt_prefix,
                vc_followup_instruction=vc_followup_instruction,
                action_max_tokens=action_max_tokens,
                action_temperature=action_temperature,
                action_stop=action_stop,
                followup_max_tokens=followup_max_tokens,
                followup_temperature=followup_temperature,
                followup_max_context_chars=followup_max_context_chars,
                followup_cot_max_chars=followup_cot_max_chars,
                vc_raw_completion_max_chars=vc_raw_completion_max_chars,
                c1_cot_temperature=c1_cot_temperature,
                c1_cot_max_tokens=c1_cot_max_tokens,
                c1_verify_temperature=c1_verify_temperature,
                c1_verify_max_tokens=c1_verify_max_tokens,
                c1_verify_stop=c1_verify_stop,
                c1_verify_instruction=c1_verify_instruction,
            )

    obs = env.reset()
    # Same as run_episode: retain full opening observation for growing prompts.
    history: list[str] = [
        f"OBSERVATION: {_truncate_for_history(obs, max_chars=history_max_obs_chars, head_ratio=history_obs_head_ratio)}"
    ]
    pinned_recipe: str | None = None
    steps = 0
    total_lm_calls = 0
    total_tokens_generated = 0
    tle_per_step: list[dict | None] = []
    vc_per_step: list[float | None] = []
    logprob_raw_per_step: list[list[dict[str, Any]] | None] = []
    vc_detail_per_step: list[dict[str, Any] | None] = []
    stage_per_step: list[str] = []
    steps_detail: list[dict[str, Any]] = []
    signal: dict[str, Any] | None = None

    trace_path: Path | None = None
    if save_step_traces:
        if not episode_id:
            warnings.warn(
                "save_step_traces is True but episode_id is missing; step traces not written."
            )
        elif not trace_output_dir:
            warnings.warn(
                "save_step_traces is True but trace_output_dir is missing; step traces not written."
            )
        else:
            trace_path = Path(trace_output_dir) / f"trace_{episode_id}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("", encoding="utf-8")

    ep_id_adapt = episode_id or "ep_unknown"
    if trace_hook is not None:
        meta_a = {"strategy": strategy, "model": trace_model_name or ""}
        try:
            trace_hook.episode_start(
                ep_id_adapt,
                metadata=meta_a,
                session_id=trace_session_id,
                tags=trace_tags,
                trace_name=trace_name,
            )
        except TypeError:
            trace_hook.episode_start(ep_id_adapt, metadata=meta_a)

    t_start = time.perf_counter()
    try:
        while not getattr(env, "done", False) and steps < max_steps:
            step_obs = obs
            step_obs_for_prompt = _truncate_for_history(
                step_obs,
                max_chars=history_current_obs_max_chars,
                head_ratio=history_obs_head_ratio,
            )
            history_snapshot = list(history)
            stage = alloc(signal, strategy, steps, rng)
            stage_per_step.append(stage)
            step_fn = resolve(stage)
            t0 = time.perf_counter()
            history_for_prompt = _compact_history_for_prompt(
                history, keep_last_pairs=history_keep_last_pairs
            )
            if pin_recipe and pinned_recipe:
                history_for_prompt = [
                    history_for_prompt[0],
                    f"PINNED RECIPE:\n{pinned_recipe}",
                    *history_for_prompt[1:],
                ]
            raw = step_fn(step_obs_for_prompt, history_for_prompt, model)
            step_wall_time_s = time.perf_counter() - t0
            action, tle, vc, tokens_used, lm_calls_this_step, log_raw, vc_det, prompt_full, response_full, call_detail = (
                normalize_step_result(raw)
            )
            tle_per_step.append(tle)
            vc_per_step.append(vc)
            if save_logprob_distributions:
                logprob_raw_per_step.append(log_raw)
            vc_detail_per_step.append(vc_det)
            total_tokens_generated += tokens_used
            total_lm_calls += lm_calls_this_step
            row_a: dict[str, Any] = {
                "step_index": steps,
                "compute_stage": stage,
                "action": action,
                "tokens_generated": int(tokens_used),
                "lm_calls_this_step": int(lm_calls_this_step),
                "step_wall_time_s": float(step_wall_time_s),
                "tle": tle,
                "vc": vc,
                "correctness": None,
                "observation_length_chars": len(step_obs or ""),
            }
            if stage == "C2" and isinstance(call_detail, dict):
                for k in ("vote_agreement", "unique_actions", "winner_index", "tie_broken"):
                    if k in call_detail:
                        row_a[k] = call_detail.get(k)
            if vc_det:
                row_a["vc_prompt"] = vc_det.get("vc_prompt")
                row_a["vc_raw_text"] = vc_det.get("vc_raw_text")
                row_a["vc_pattern_matched"] = vc_det.get("vc_pattern_matched")
                row_a["vc_tokens_used"] = vc_det.get("vc_tokens_used")
                if save_vc_distributions and vc_det.get("vc_logprobs") is not None:
                    row_a["vc_logprobs"] = vc_det.get("vc_logprobs")
            steps_detail.append(row_a)
            obs = env.step(action)
            if pin_recipe and pinned_recipe is None:
                rb = _extract_recipe_block(obs)
                if rb:
                    pinned_recipe = rb
            correctness_ad: str | None = None
            sr_a = getattr(env, "step_results", None)
            if isinstance(sr_a, list) and sr_a:
                correctness_ad = sr_a[-1].get("correctness")  # type: ignore[union-attr]
            if trace_path is not None:
                trace_rec_a: dict[str, Any] = {
                    "step_index": steps,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "compute_stage": stage,
                    "strategy": strategy,
                    "prompt_full": prompt_full or "",
                    "response_full": response_full or "",
                    "call_detail": call_detail or None,
                    "action_parsed": action,
                    "observation_before": step_obs,
                    "observation_after": obs,
                    "history_snapshot": history_snapshot,
                    "tle": tle,
                    "vc": vc,
                    "correctness": correctness_ad,
                    "step_wall_time_s": float(step_wall_time_s),
                    "lm_calls": int(lm_calls_this_step),
                    "tokens_generated": int(tokens_used),
                }
                if isinstance(call_detail, dict):
                    subcalls = call_detail.get("subcalls")
                    if isinstance(subcalls, list):
                        for sc in subcalls:
                            if not isinstance(sc, dict):
                                continue
                            kind = str(sc.get("kind") or "").strip().lower()
                            if kind == "cot":
                                trace_rec_a["cot_prompt"] = sc.get("prompt") or ""
                                trace_rec_a["cot_response"] = sc.get("response") or ""
                            elif kind == "verify":
                                trace_rec_a["verify_prompt"] = sc.get("prompt") or ""
                                trace_rec_a["verify_response"] = sc.get("response") or ""
                if vc_det:
                    trace_rec_a["vc_followup_prompt"] = vc_det.get("vc_prompt")
                    trace_rec_a["vc_followup_response"] = vc_det.get("vc_raw_text")
                write_step_trace_line(trace_path, trace_rec_a)
            if trace_hook is not None:
                lf_meta_a: dict[str, Any] = {
                    "tle": tle,
                    "vc": vc,
                    "correctness": correctness_ad,
                    "strategy": strategy,
                    "tokens_generated": int(tokens_used),
                    "lm_calls": int(lm_calls_this_step),
                }
                subcalls = None
                if isinstance(call_detail, dict) and isinstance(call_detail.get("subcalls"), list):
                    subcalls = call_detail.get("subcalls")
                log_step_with_fallback(
                    trace_hook=trace_hook,
                    step_index=steps,
                    stage=stage,
                    strategy=strategy,
                    observation=step_obs,
                    action=action,
                    prompt_full=prompt_full,
                    response_full=response_full,
                    vc_det=vc_det,
                    trace_model_name=trace_model_name,
                    metadata=lf_meta_a,
                    subcalls=subcalls,
                )
            # Maintain growing context as ACTION/OBSERVATION pairs.
            history.append(f"ACTION: {action}")
            history.append(
                f"OBSERVATION: {_truncate_for_history(obs, max_chars=history_max_obs_chars, head_ratio=history_obs_head_ratio)}"
            )
            steps += 1
            signal = _signal_for_next_step(tle, vc)
            if on_step is not None:
                on_step(
                    {
                        "step_index": steps - 1,
                        "episode_steps": steps,
                        "max_steps": max_steps,
                        "env_done": bool(getattr(env, "done", False)),
                        "compute_stage": stage,
                        "strategy": strategy,
                        "lm_calls_this_step": int(lm_calls_this_step),
                        "total_lm_calls": int(total_lm_calls),
                    }
                )
    finally:
        if trace_hook is not None:
            try:
                succ = bool(getattr(env, "task_success", False)) or bool(
                    getattr(env, "done", False)
                )
            except Exception:
                succ = False
            final_tags: list[str] = []
            if succ:
                final_tags.append("succeeded")
            else:
                final_tags.append("failed")
            if steps >= max_steps:
                final_tags.append("hit_step_cap")
            try:
                trace_hook.episode_end(
                    output={
                        "task_success": succ,
                        "steps": int(steps),
                        "total_lm_calls": int(total_lm_calls),
                        "total_tokens_generated": int(total_tokens_generated),
                    },
                    final_tags=final_tags,
                )
            except TypeError:
                trace_hook.episode_end()

    wall_clock_time = time.perf_counter() - t_start
    if hasattr(env, "task_success"):
        task_success = bool(getattr(env, "task_success"))
    else:
        task_success = bool(getattr(env, "done", False))
    step_correctness = _copy_step_results(env)
    if step_correctness is not None:
        by_idx: dict[int, dict[str, Any]] = {}
        for d in step_correctness:
            try:
                idx = int(d.get("step_index"))
            except Exception:
                continue
            by_idx[idx] = d
        for sd in steps_detail:
            idx = int(sd["step_index"])
            corr = by_idx.get(idx, {}).get("correctness")
            sd["correctness"] = corr if corr is not None else None
    normalized_compute_cost = (
        (total_lm_calls / (max_steps * 3)) if max_steps and (max_steps * 3) > 0 else 0.0
    )
    efficiency_score = (
        (float(task_success) / normalized_compute_cost) if normalized_compute_cost > 0 else None
    )
    adaptive_out: dict[str, Any] = {
        "steps": steps,  # legacy
        "episode_length_steps": steps,
        "task_success": task_success,
        "lm_calls": steps,  # legacy
        "total_lm_calls": int(total_lm_calls),
        "tokens": int(total_tokens_generated),  # legacy
        "total_tokens_generated": int(total_tokens_generated),
        "normalized_compute_cost": float(normalized_compute_cost),
        "efficiency_score": efficiency_score,
        "wall_clock_time": wall_clock_time,
        "tle_per_step": tle_per_step,
        "vc_per_step": vc_per_step,
        "step_correctness": step_correctness,
        "stage_per_step": stage_per_step,
        "steps_detail": steps_detail,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if save_logprob_distributions:
        adaptive_out["logprob_raw_per_step"] = logprob_raw_per_step
    if any(x is not None for x in vc_detail_per_step):
        adaptive_out["vc_detail_per_step"] = vc_detail_per_step
    return adaptive_out
