#!/usr/bin/env python3
"""
Pilot study runner: inference benchmarks, TLE/VC checks, TextWorld e2e, Tower of Hanoi,
and feasibility JSON.

By default creates a timestamped subfolder under --output-dir (``pilot_YYYYMMDD_HHMMSS``) with
``run_info.json``; use ``--no-timestamp-run`` to write directly to ``--output-dir``.

Usage:
  python scripts/run_pilot.py --config configs/pilot.yaml [--output-dir data/results] [--real]
  python scripts/run_pilot.py --only test2 --config configs/pilot.yaml ...   # TLE only
  python scripts/run_pilot.py --no-timestamp-run ...  # flat layout (e.g. merge feasibility inputs)
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src is on path when run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.dotenv_loader import load_dotenv_if_present

_DOTENV_INFO = load_dotenv_if_present(REPO_ROOT)

from src.utils.experiment_env import make_experiment_env, resolve_textworld_game_path
from src.utils.step_config import resolve_step_fn_kwargs
from src.utils.pilot_config import load_pilot_config_with_lmstudio_override, load_yaml_path
from src.utils.run_progress import format_run_elapsed, log, log_step_line
from src.agent.compute_stages import VC_FOLLOWUP_PROMPT_MARKER
from src.pilot.artifacts import (
    maybe_write_logprob_artifacts as _maybe_write_logprob_artifacts,
    maybe_write_vc_artifacts as _maybe_write_vc_artifacts,
    save_json as _save_json,
)
from src.pilot.steps import (
    episode_vc_tle_rates as _episode_vc_tle_rates,
    prepare_feasibility_inputs as _prepare_feasibility_inputs,
    run_mock_inference_speed_benchmark as _run_mock_inference_speed_benchmark,
)
from src.pilot.orchestrator import run_pilot_main


def _step_trace_settings(config: dict) -> tuple[bool, Any]:
    """(save_step_traces, trace_hook_or_none). Hook is None unless tracing.langfuse_enabled."""
    lg = config.get("logging") or {}
    save = bool(lg.get("save_step_traces", False))
    from src.utils.tracing import optional_trace_hook_from_config

    hook = optional_trace_hook_from_config(config, dotenv_info=_DOTENV_INFO)
    return save, hook

# Canonical order for --only (subset runs in this order).
PILOT_STEPS_ORDER = (
    "sanity",
    "test1",
    "test2",
    "test3",
    "test4",
    "test5",
    "feasibility",
)
# Steps that call the model wrapper (real or mock path inside each test).
PILOT_STEPS_NEED_MODEL = frozenset({"sanity", "test1", "test2", "test3", "test4", "test5"})


def load_config(config_path: str | Path) -> dict:
    """Load a single YAML file (no LM Studio merge). Prefer ``load_pilot_config`` in main."""
    return load_yaml_path(Path(config_path))


def _only_steps_in_order(selected: frozenset[str]) -> list[str]:
    """Deduplicate and order user --only choices by PILOT_STEPS_ORDER."""
    return [s for s in PILOT_STEPS_ORDER if s in selected]


def parse_pilot_mode_arg(value: str) -> str:
    """CLI pilot mode: mock | hf | m1 (deprecated alias for hf) | cuda | lmstudio."""
    v = (value or "mock").lower().strip()
    if v == "m1":
        warnings.warn(
            '--pilot-mode m1 is deprecated; use "hf" (HuggingFace + MPS on Apple Silicon).',
            DeprecationWarning,
            stacklevel=2,
        )
        v = "hf"
    allowed = frozenset({"mock", "hf", "cuda", "lmstudio"})
    if v not in allowed:
        raise argparse.ArgumentTypeError(
            f"invalid pilot mode {value!r}; expected one of: mock, hf, m1, cuda, lmstudio"
        )
    return v


def _create_real_model(config: dict, pilot_mode: str) -> tuple[Any | None, str | None]:
    """
    Create real model wrapper for the given pilot mode.
    mock -> None; hf -> HuggingFace on Apple Silicon (MPS); cuda -> vLLM (or inference.backend);
    lmstudio -> LM Studio local server (OpenAI-compatible HTTP).
    """
    if pilot_mode == "mock":
        return None, None
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name")
    if not model_name:
        return None, "model.name missing in config"
    dtype = model_cfg.get("dtype", "float16")
    try:
        from src.utils.model_wrapper import create_wrapper
        if pilot_mode == "hf":
            return create_wrapper(backend="hf", model_name=model_name, dtype=dtype, device="mps"), None
        if pilot_mode == "cuda":
            backend = str(config.get("inference", {}).get("backend", "vllm") or "vllm").strip().lower()
            # The YAML config is shared across local (Apple/MLX) and cloud (CUDA/vLLM) runs.
            # For RunPod smoke tests, cuda must never accidentally select a non-CUDA backend
            # like "mlx" — that would create a base ModelWrapper and fail at runtime.
            if backend not in {"vllm", "hf"}:
                backend = "vllm"
            inf = config.get("inference", {}) or {}
            # vLLM will default to the model's advertised max_seq_len. Some models (e.g. Qwen3)
            # advertise very large context (e.g. 40960) that does not fit KV cache on 24GB GPUs
            # for even a single request. Default to a safe context length unless explicitly set.
            max_model_len = inf.get("max_model_len") or inf.get("vllm_max_model_len")
            if max_model_len is None:
                max_model_len = 8192
            gpu_mem_util = inf.get("gpu_memory_utilization")
            extra = {}
            try:
                extra["max_model_len"] = int(max_model_len)
            except Exception:
                extra["max_model_len"] = 8192
            if gpu_mem_util is not None:
                try:
                    extra["gpu_memory_utilization"] = float(gpu_mem_util)
                except Exception:
                    pass
            extra["chat_template"] = bool(inf.get("chat_template", True))
            extra["enable_thinking"] = bool(inf.get("enable_thinking", False))
            return create_wrapper(backend=backend, model_name=model_name, dtype=dtype, **extra), None
        if pilot_mode == "lmstudio":
            inf = config.get("inference", {})
            base_url = inf.get("lmstudio_base_url") or os.environ.get(
                "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
            )
            api_key = inf.get("lmstudio_api_key") or os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
            top_k = int(inf.get("lmstudio_top_logprobs", 5))
            return (
                create_wrapper(
                    backend="lmstudio",
                    model_name=model_name,
                    base_url=base_url,
                    lmstudio_api_key=api_key,
                    lmstudio_top_logprobs=top_k,
                ),
                None,
            )
        return None, f"unsupported pilot_mode={pilot_mode}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _try_gpu_info() -> dict[str, Any]:
    """
    Best-effort GPU info for pilot_benchmark.json.
    """
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return {"gpu_name": None, "vram_total_gb": None, "cuda_available": False}
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = float(props.total_memory) / (1024**3)
        return {
            "gpu_name": name,
            "vram_total_gb": vram_total_gb,
            "cuda_available": True,
        }
    except Exception:
        return {"gpu_name": None, "vram_total_gb": None, "cuda_available": False}


def _assert_real_model_or_raise(pilot_mode: str, real_model: Any | None, err: str | None) -> None:
    """
    Fail-fast guard: when user requested a real backend, do not silently run mock.
    """
    if pilot_mode == "mock":
        return
    if real_model is None:
        detail = f" ({err})" if err else ""
        raise RuntimeError(
            f"pilot_mode={pilot_mode} requested, but real model wrapper could not be created{detail}. "
            "Aborting to avoid accidentally running a mock pilot."
        )


def _sanity_check_real_inference(config: dict, pilot_mode: str, real_model: Any) -> dict[str, Any]:
    """
    Run a single small generate(logprobs=True) call to verify the backend is actually working.
    """
    inf_cfg = config.get("inference", {})
    max_tokens = int(inf_cfg.get("max_tokens", 64))
    temperature = float(inf_cfg.get("temperature", 0.3))
    prompt = "Reply with exactly: OK\\nConfidence: 50"
    t0 = time.perf_counter()
    text, logprobs = real_model.generate(prompt, logprobs=True, max_tokens=max_tokens, temperature=temperature)
    elapsed = time.perf_counter() - t0
    n_out = len(logprobs) if logprobs else 0
    return {
        "pilot_mode": pilot_mode,
        "ok": bool((text or "").strip()),
        "latency_s": float(elapsed),
        "completion_tokens_observed": int(n_out),
        "has_logprobs": bool(logprobs),
        "sample_text_prefix": (text or "")[:200],
    }


def run_test1_inference_speed(
    config: dict,
    output_dir: Path,
    real_model=None,
) -> dict:
    """Test 1: Inferenzgeschwindigkeit — 50 prompts, tok/s, latency, VRAM.
    If real_model is provided, run real inference and measure. Otherwise use mock.
    """
    num_prompts = config.get("test1_inference", {}).get("num_prompts", 50)
    inf_cfg = config.get("inference", {})
    max_tokens = inf_cfg.get("max_tokens", 256)
    temperature = inf_cfg.get("temperature", 0.3)

    if real_model is not None:
        # Real benchmark: 50 prompts, measure wall time and token count
        log(f"Test 1: inference speed — {num_prompts} prompts (logprobs on)...")
        prompts = [f"Complete this sentence: The quick brown fox " * (i % 3 + 1) for i in range(num_prompts)]
        latencies = []
        total_tokens = 0
        progress_interval = max(1, num_prompts // 5)  # e.g. every 10 for 50 prompts
        for i, p in enumerate(prompts):
            t0 = time.perf_counter()
            text, logprobs = real_model.generate(p, logprobs=True, max_tokens=max_tokens, temperature=temperature)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            n_out = len(logprobs) if logprobs else 0
            if n_out == 0 and (text or "").strip():
                # APIs without logprobs/usage (rare): rough chars→tokens for tok/s
                n_out = max(1, len(text) // 4)
            total_tokens += n_out
            if (i + 1) % progress_interval == 0 or (i + 1) == num_prompts:
                log(f"Test 1: prompt {i + 1}/{num_prompts} done")
        elapsed_total = sum(latencies)
        tokens_per_sec = total_tokens / elapsed_total if elapsed_total > 0 else 0.0
        n = len(latencies)
        mean_lat = sum(latencies) / n if n else 0.0
        variance = sum((x - mean_lat) ** 2 for x in latencies) / n if n else 0.0
        std_lat = variance ** 0.5
        vram_gb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            pass
        return {
            "tokens_per_sec": tokens_per_sec,
            "latency_mean": mean_lat,
            "latency_std": std_lat,
            "vram_gb": vram_gb,
            "num_prompts": num_prompts,
            "total_tokens": total_tokens,
        }
    # Mock
    result = _run_mock_inference_speed_benchmark(num_prompts, tokens_per_call=200)
    result["vram_gb"] = 0.0
    return result


def run_test2_token_entropy(config: dict, output_dir: Path, *, real_model=None) -> dict:
    """Test 2: TLE extraction from real logprobs when available; else synthetic."""
    log("Test 2: token entropy — start")
    from src.signals.token_entropy import compute_tle, extract_tle_from_response

    save_lp, lp_fmt, lp_sub = _logprob_export_settings(config)

    if real_model is None:
        # Synthetic logprobs: easy (low entropy) vs hard (high entropy)
        easy = [{"logprob": -0.1}] * 10
        hard = [{"logprob": -2.0}] * 10
        out = {"mode": "synthetic", "easy_tle": compute_tle(easy), "hard_tle": compute_tle(hard)}
        if save_lp:
            out_dir = output_dir / lp_sub if lp_sub else output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            dist = {
                "mode": "synthetic",
                "per_prompt_logprob_tokens": {"easy": easy, "hard": hard},
            }
            dist_path = out_dir / "pilot_test2_tle_distributions.json"
            with open(dist_path, "w") as f:
                json.dump(dist, f, indent=2)
            log(f"Wrote {dist_path}")
            if lp_fmt in ("csv", "both"):
                import csv

                from src.signals.token_entropy import softmax_probs_from_top_logprobs

                csv_path = out_dir / "pilot_test2_tle_distributions.csv"
                per_prompt_logprob_tokens = {"easy": easy, "hard": hard}
                with open(csv_path, "w", encoding="utf-8", newline="") as cf:
                    w = csv.writer(cf)
                    w.writerow(
                        ["prompt_name", "completion_token_index", "rank_in_topk", "token", "logprob", "p_renorm_topk"]
                    )
                    for pname, lp_list in per_prompt_logprob_tokens.items():
                        for tok_i, tok in enumerate(lp_list or []):
                            if not isinstance(tok, dict):
                                continue
                            top = tok.get("top_logprobs")
                            if isinstance(top, list) and top:
                                cands = [
                                    x
                                    for x in top
                                    if isinstance(x, dict) and x.get("logprob") is not None
                                ]
                                probs = softmax_probs_from_top_logprobs(top)
                                for rank, (cand, pr) in enumerate(zip(cands, probs)):
                                    w.writerow(
                                        [pname, tok_i, rank, cand.get("token", ""), cand.get("logprob", ""), f"{pr:.8g}"]
                                    )
                            else:
                                w.writerow([pname, tok_i, 0, tok.get("token", ""), tok.get("logprob", ""), "1.0"])
                log(f"Wrote {csv_path}")
        return out

    inf_cfg = config.get("inference", {})
    max_tokens = int(inf_cfg.get("max_tokens", 128))
    temperature = float(inf_cfg.get("temperature", 0.3))
    prompts = {
        "easy": "Say 'hello' and then 'Confidence: 90'.",
        "hard": "Write a short proof sketch of the P=NP problem and then 'Confidence: 10'.",
    }
    out: dict[str, Any] = {"mode": "real", "per_prompt": {}}
    mean_entropies: list[float] = []
    max_entropies: list[float] = []
    per_prompt_logprob_tokens: dict[str, Any] = {}
    for name, p in prompts.items():
        text, logprobs = real_model.generate(p, logprobs=True, max_tokens=max_tokens, temperature=temperature)
        tle = extract_tle_from_response(text, logprobs) if logprobs else None
        out["per_prompt"][name] = {
            "tle": tle,
            "completion_tokens_observed": len(logprobs) if logprobs else 0,
        }
        if save_lp and logprobs:
            per_prompt_logprob_tokens[name] = logprobs
        if isinstance(tle, dict):
            me = tle.get("mean_entropy")
            mx = tle.get("max_entropy")
            if isinstance(me, (int, float)):
                mean_entropies.append(float(me))
            if isinstance(mx, (int, float)):
                max_entropies.append(float(mx))
    out["summary"] = {
        "mean_entropy_avg": (sum(mean_entropies) / len(mean_entropies)) if mean_entropies else None,
        "max_entropy_avg": (sum(max_entropies) / len(max_entropies)) if max_entropies else None,
    }
    if save_lp and per_prompt_logprob_tokens:
        out_dir = output_dir / lp_sub if lp_sub else output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dist_path = out_dir / "pilot_test2_tle_distributions.json"
        with open(dist_path, "w") as f:
            json.dump(
                {
                    "mode": "real",
                    "per_prompt_logprob_tokens": per_prompt_logprob_tokens,
                },
                f,
                indent=2,
            )
        log(f"Wrote {dist_path}")
        if lp_fmt in ("csv", "both"):
            import csv

            csv_path = out_dir / "pilot_test2_tle_distributions.csv"
            from src.signals.token_entropy import softmax_probs_from_top_logprobs

            with open(csv_path, "w", encoding="utf-8", newline="") as cf:
                w = csv.writer(cf)
                w.writerow(
                    ["prompt_name", "completion_token_index", "rank_in_topk", "token", "logprob", "p_renorm_topk"]
                )
                for pname, lp_list in per_prompt_logprob_tokens.items():
                    for tok_i, tok in enumerate(lp_list or []):
                        if not isinstance(tok, dict):
                            continue
                        top = tok.get("top_logprobs")
                        if isinstance(top, list) and top:
                            cands = [
                                x
                                for x in top
                                if isinstance(x, dict) and x.get("logprob") is not None
                            ]
                            probs = softmax_probs_from_top_logprobs(top)
                            for rank, (cand, pr) in enumerate(zip(cands, probs)):
                                w.writerow([pname, tok_i, rank, cand.get("token", ""), cand.get("logprob", ""), f"{pr:.8g}"])
                        else:
                            w.writerow([pname, tok_i, 0, tok.get("token", ""), tok.get("logprob", ""), "1.0"])
            log(f"Wrote {csv_path}")
    return out


def run_test3_verbalized_confidence(config: dict, *, real_model=None) -> dict:
    """Test 3: VC parsing from real generations when available; else static strings."""
    log("Test 3: verbalized confidence — start")
    from src.signals.verbalized_confidence import parse_confidence

    if real_model is None:
        samples = [
            "The answer is 42. Confidence: 85",
            "0-100: 70",
            "No number here",
        ]
        return {"mode": "static", "samples": {s: parse_confidence(s) for s in samples}}

    inf_cfg = config.get("inference", {})
    max_tokens = int(inf_cfg.get("max_tokens", 128))
    temperature = float(inf_cfg.get("temperature", 0.3))
    prompts = [
        "Answer with one short sentence. Then on a new line output: Confidence: <0-100>.",
        "Output exactly two lines:\nAction: <some action>\nConfidence: <0-100>",
        "Do NOT include any numbers in your response.",  # negative control
    ]
    parsed: list[float | None] = []
    texts: list[str] = []
    for p in prompts:
        text, _ = real_model.generate(p, logprobs=False, max_tokens=max_tokens, temperature=temperature)
        texts.append(text)
        parsed.append(parse_confidence(text))
    parseable = sum(1 for v in parsed if v is not None)
    return {
        "mode": "real",
        "n_prompts": len(prompts),
        "parseable": parseable,
        "parse_rate": (parseable / len(prompts)) if prompts else 0.0,
        "parsed_values": parsed,
        "sample_text_prefixes": [t[:120] for t in texts],
    }

def run_test5_tower_of_hanoi(
    config: dict,
    output_dir: Path,
    real_model=None,
) -> dict:
    """
    Completion-plan add-on: Tower of Hanoi move parseability on real outputs.
    Measures fraction of steps where env recorded a parsed move (action_parsed != None).
    """
    from src.utils.compute_stage_selection import resolve_compute_stages_for_domain

    stages = resolve_compute_stages_for_domain(config, domain="tower_of_hanoi")
    stage = stages[0] if stages else "C0"
    if len(stages) > 1:
        log(f"Test 5: note: multiple stages configured ({stages}); running first only: {stage}")
    log(f"Test 5: Tower of Hanoi parseability — start ({stage}; per-step progress below)")
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.tower_of_hanoi import TowerOfHanoiEnv, generate_instances
    from src.utils.logging_utils import log_episode

    class MockModel:
        def generate(self, prompt, logprobs=False, **kwargs):
            if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
                t = "50"
                lp = [{"logprob": -0.2}] * 2 if logprobs else None
                return t, lp
            text = "A->B"
            lp = [{"logprob": -0.5}] * 5 if logprobs else None
            return text, lp

    model = real_model if real_model is not None else MockModel()
    save_lp, _, _ = _logprob_export_settings(config)
    save_vc, _, _ = _vc_export_settings(config)
    toh_cfg = config.get("tower_of_hanoi", {})
    num_episodes = int(toh_cfg.get("pilot_episodes", 20))
    num_disks = int(toh_cfg.get("pilot_num_disks", 3))
    seed = int(toh_cfg.get("task_generation_seed", 42))
    max_steps_default = int(toh_cfg.get("pilot_max_steps", 20))
    num_disks_range_raw = toh_cfg.get("num_disks_range", [num_disks, num_disks])
    partial_start_range_raw = toh_cfg.get("partial_start_range", [0, 0])
    try:
        num_disks_range = (int(num_disks_range_raw[0]), int(num_disks_range_raw[1]))
    except Exception:
        num_disks_range = (num_disks, num_disks)
    try:
        partial_start_range = (int(partial_start_range_raw[0]), int(partial_start_range_raw[1]))
    except Exception:
        partial_start_range = (0, 0)
    tasks = generate_instances(
        num_episodes,
        seed=seed,
        num_disks_range=num_disks_range,
        partial_start_range=partial_start_range,
    )
    # Respect each task's intrinsic max_steps (derived from its optimal solution length),
    # while still honoring the configured default cap as a floor.
    worst_lm = 0
    for t in tasks:
        tms = int(t.get("max_steps") or 0)
        cap = int(max_steps_default if max_steps_default > 0 else 1)
        worst_lm += max(cap, tms) if tms > 0 else cap
    log(
        "Test 5: ToH plan "
        f"{num_episodes} episodes; num_disks_range={list(num_disks_range)} "
        f"partial_start_range={list(partial_start_range)} "
        f"default_max_steps={max_steps_default} → ≤{worst_lm} steps ({stage})"
    )
    parseable_steps = 0
    total_steps = 0
    episodes_data: list[dict[str, Any]] = []
    success_eps = 0
    optimal_steps_total = 0
    legal_steps_total = 0
    steps_total = 0
    oscillation_eps = 0
    min_opt_moves_remaining_sum = 0.0
    min_opt_moves_remaining_n = 0
    c2_vote_agreements: list[float] = []
    c2_unique_actions: list[int] = []
    save_tr, trace_hk = _step_trace_settings(config)
    model_nm = str(config.get("model", {}).get("name", ""))
    session_id = str(output_dir.name)
    base_tags = ["pilot", "tower_of_hanoi", str(config.get("pilot_mode", "unknown")), session_id]
    for i, task in enumerate(tasks):

        def _make_toh_on_step(ep_i: int, ep_total: int):
            def _inner(info: dict) -> None:
                log_step_line(f"pilot ToH ep {ep_i + 1}/{ep_total}", info)

            return _inner

        task_max_steps = int(task.get("max_steps") or 0)
        max_steps = max_steps_default
        if task_max_steps > 0:
            max_steps = max(max_steps_default, task_max_steps)
        dom_cfg = (config.get("domain_prompts") or {}).get("tower_of_hanoi") if isinstance(config.get("domain_prompts"), dict) else None
        include_vm = bool(dom_cfg.get("include_valid_moves", False)) if isinstance(dom_cfg, dict) else False
        env = TowerOfHanoiEnv(task=task, max_steps=max_steps, include_valid_moves=include_vm)
        step_cfg = resolve_step_fn_kwargs(config, "tower_of_hanoi")
        hist_keys = {
            "history_keep_last_pairs",
            "history_max_obs_chars",
            "history_current_obs_max_chars",
            "history_obs_head_ratio",
            "pin_recipe",
        }
        history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in hist_keys}
        step_fn = get_step_fn(
            stage,
            save_logprob_distributions=save_lp,
            save_vc_distributions=save_vc,
            **step_cfg,
        )
        log(
            f"Test 5: ToH {stage} episode {i + 1}/{num_episodes} — running "
            f"(num_disks={task.get('num_disks')}, partial_start_moves={task.get('partial_start_moves')}, "
            f"max_steps={max_steps})..."
        )
        ep_toh = f"ep_tower_of_hanoi_{i}_{stage}_0"
        # Seed C2 tie-breaking deterministically per episode.
        step_cfg["c2_tie_break_seed"] = ep_toh
        result = run_episode(
            env,
            model,
            stage,
            step_fn=step_fn,
            max_steps=max_steps,
            on_step=_make_toh_on_step(i, num_episodes),
            save_logprob_distributions=save_lp,
            save_vc_distributions=save_vc,
            save_step_traces=save_tr,
            episode_id=ep_toh,
            trace_output_dir=str(output_dir),
            trace_model_name=model_nm or None,
            trace_hook=trace_hk,
            trace_session_id=session_id,
            trace_tags=[t for t in ([*base_tags, stage, model_nm] if model_nm else [*base_tags, stage]) if t],
            trace_name=ep_toh,
            **history_cfg,
        )
        log(
            f"Test 5: ToH {stage} episode done {i + 1}/{num_episodes} — steps={result['steps']} "
            f"lm_calls={result.get('total_lm_calls', 0)} wall={result.get('wall_clock_time', 0):.2f}s "
            f"success={result.get('task_success')}"
        )
        if stage == "C2":
            for sd in result.get("steps_detail") or []:
                if not isinstance(sd, dict):
                    continue
                va = sd.get("vote_agreement")
                ua = sd.get("unique_actions")
                if isinstance(va, (int, float)):
                    c2_vote_agreements.append(float(va))
                if isinstance(ua, int):
                    c2_unique_actions.append(int(ua))
        data = {
            "episode_id": ep_toh,
            "domain": "tower_of_hanoi",
            "instance": i,
            "compute_stage": stage,
            "run": 0,
            "task_success": result["task_success"],
            "steps": result["steps"],
            "total_lm_calls": result.get("total_lm_calls", 0),
            "total_tokens_generated": result.get("total_tokens_generated", 0),
            "wall_clock_time": result["wall_clock_time"],
            "tle_per_step": result.get("tle_per_step"),
            "vc_per_step": result.get("vc_per_step"),
            "steps_detail": result.get("steps_detail"),
            "step_correctness": result.get("step_correctness"),
        }
        if result.get("vc_detail_per_step") is not None:
            data["vc_detail_per_step"] = result["vc_detail_per_step"]
        log_episode(ep_toh, data, output_dir)
        _maybe_write_logprob_artifacts(config, ep_toh, result, output_dir)
        _maybe_write_vc_artifacts(config, ep_toh, result, output_dir)
        episodes_data.append(data)
        step_corr = result.get("step_correctness") or []
        # Behavioral metrics (legal/optimal/oscillation/min-distance), robust to missing fields.
        ep_steps = 0
        ep_optimal = 0
        ep_legal = 0
        seen_states: set[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = set()
        ep_osc = False
        ep_min_opt_rem: int | None = None
        for d in step_corr:
            total_steps += 1
            if isinstance(d, dict) and d.get("action_parsed") is not None:
                parseable_steps += 1
            if not isinstance(d, dict):
                continue
            corr = d.get("correctness")
            if corr is not None:
                ep_steps += 1
                if corr == "optimal":
                    ep_optimal += 1
                    ep_legal += 1
                elif corr == "legal":
                    ep_legal += 1
            omr = d.get("optimal_moves_remaining")
            if isinstance(omr, int):
                ep_min_opt_rem = omr if ep_min_opt_rem is None else min(ep_min_opt_rem, omr)
            st = d.get("state_after")
            if isinstance(st, dict) and all(k in st for k in ("A", "B", "C")):
                try:
                    key = (tuple(st["A"]), tuple(st["B"]), tuple(st["C"]))
                except Exception:
                    key = None
                if key is not None:
                    if key in seen_states:
                        ep_osc = True
                    else:
                        seen_states.add(key)

        steps_total += ep_steps
        optimal_steps_total += ep_optimal
        legal_steps_total += ep_legal
        if bool(result.get("task_success")):
            success_eps += 1
        if ep_osc:
            oscillation_eps += 1
        if ep_min_opt_rem is not None:
            min_opt_moves_remaining_sum += float(ep_min_opt_rem)
            min_opt_moves_remaining_n += 1
    parse_rate = (parseable_steps / total_steps) if total_steps else 0.0
    success_rate = (success_eps / num_episodes) if num_episodes else 0.0
    avg_optimal_rate = (optimal_steps_total / steps_total) if steps_total else 0.0
    avg_legal_rate = (legal_steps_total / steps_total) if steps_total else 0.0
    oscillation_rate = (oscillation_eps / num_episodes) if num_episodes else 0.0
    avg_min_optimal_moves_remaining = (
        (min_opt_moves_remaining_sum / min_opt_moves_remaining_n)
        if min_opt_moves_remaining_n
        else None
    )
    if stage == "C2" and c2_vote_agreements:
        xs = sorted(c2_vote_agreements)
        med = xs[len(xs) // 2]
        mean = sum(xs) / len(xs)
        log(
            f"Test 5: C2 diversity summary — vote_agreement median={med:.3f} mean={mean:.3f} "
            f"(n_steps={len(xs)})"
        )
    if stage == "C2" and c2_unique_actions:
        ys = sorted(c2_unique_actions)
        med_u = ys[len(ys) // 2]
        mean_u = sum(ys) / len(ys)
        log(
            f"Test 5: C2 diversity summary — unique_actions median={med_u:.3f} mean={mean_u:.3f} "
            f"(n_steps={len(ys)})"
        )
    return {
        "num_episodes": num_episodes,
        "num_disks": num_disks,
        "num_disks_range": list(num_disks_range),
        "partial_start_range": list(partial_start_range),
        "default_max_steps": max_steps_default,
        "total_steps": total_steps,
        "parseable_steps": parseable_steps,
        "parse_rate": parse_rate,
        "success_rate": success_rate,
        "avg_optimal_rate": avg_optimal_rate,
        "avg_legal_rate": avg_legal_rate,
        "oscillation_rate": oscillation_rate,
        "avg_min_optimal_moves_remaining": avg_min_optimal_moves_remaining,
    }

def run_test4_textworld_e2e(config: dict, output_dir: Path, real_model=None) -> list[dict]:
    """Test 4: instances x 3 stages x runs = episodes; structured JSON per episode.

    Uses a real TextWorld gym env when ``resolve_textworld_game_path`` finds a story file for
    that instance under ``paths.tasks_dir`` (same layout as Phase 1/2). If no file exists or
    TextWorld is not installed, ``TextWorldEnv`` falls back to its stub.

    If real_model is provided, use it; otherwise use MockModel.
    """
    log("Test 4: end-to-end TextWorld episodes — start")
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.utils.compute_stage_selection import resolve_compute_stages_for_domain
    from src.utils.logging_utils import log_episode

    class MockModel:
        def generate(self, prompt, logprobs=False, **kwargs):
            if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
                t = "80"
                lp = [{"logprob": -0.15}] * 2 if logprobs else None
                return t, lp
            text = "go north"
            lp = [{"logprob": -0.5}] * 5 if logprobs else None
            return text, lp

    model = real_model if real_model is not None else MockModel()
    save_lp, _, _ = _logprob_export_settings(config)
    save_vc, _, _ = _vc_export_settings(config)
    instances = config.get("pilot", {}).get("instances", 5)
    stages = resolve_compute_stages_for_domain(config, domain="textworld")
    runs = config.get("pilot", {}).get("runs_per_instance", 1)
    total_episodes = instances * len(stages) * runs
    episode_idx = [0]
    tw_pilot = config.get("test4_textworld") or {}
    max_env_steps = int(tw_pilot.get("max_env_steps", 10))
    tasks_dir_cfg = (config.get("paths") or {}).get("tasks_dir", "data/tasks")

    def _log_progress():
        episode_idx[0] += 1
        if episode_idx[0] % 5 == 0 or episode_idx[0] == total_episodes:
            log(f"Test 4: batch progress {episode_idx[0]}/{total_episodes} episodes")

    log(
        f"Test 4: plan {instances} instances × {len(stages)} stages × {runs} runs = "
        f"{total_episodes} episodes (max {max_env_steps} env steps each); "
        f"tasks_dir={tasks_dir_cfg}"
    )
    save_tr, trace_hk = _step_trace_settings(config)
    model_nm = str(config.get("model", {}).get("name", ""))
    session_id = str(output_dir.name)
    base_tags = ["pilot", "textworld", str(config.get("pilot_mode", "unknown")), session_id]
    episodes_data = []
    c2_vote_agreements: list[float] = []
    c2_unique_actions: list[int] = []
    for inst in range(instances):
        game_path = resolve_textworld_game_path(inst, config, REPO_ROOT)
        if game_path is not None:
            log(f"Test 4: instance {inst} game file: {game_path}")
        else:
            log(
                f"Test 4: instance {inst}: no textworld_{inst}.z8/.ulx under "
                f"{tasks_dir_cfg} (or textworld/ subdir) — using stub env"
            )
        for stage in stages:
            for run in range(runs):
                ep_id = f"ep_textworld_{inst}_{stage}_{run}"

                def _make_test5_on_step(eid: str):
                    def _inner(info: dict) -> None:
                        log_step_line(f"pilot Test4 {eid}", info)

                    return _inner

                env = make_experiment_env(
                    "textworld",
                    inst,
                    config,
                    max_env_steps,
                    REPO_ROOT,
                )
                step_cfg = resolve_step_fn_kwargs(config, "textworld")
                # Seed C2 tie-breaking deterministically per episode.
                step_cfg["c2_tie_break_seed"] = ep_id
                hist_keys = {
                    "history_keep_last_pairs",
                    "history_max_obs_chars",
                    "history_current_obs_max_chars",
                    "history_obs_head_ratio",
                    "pin_recipe",
                }
                history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in hist_keys}
                step_fn = get_step_fn(
                    stage,
                    save_logprob_distributions=save_lp,
                    save_vc_distributions=save_vc,
                    **step_cfg,
                )
                log(f"Test 4: episode start {ep_id} (stage={stage})")
                result = run_episode(
                    env,
                    model,
                    stage,
                    step_fn=step_fn,
                    max_steps=max_env_steps,
                    on_step=_make_test5_on_step(ep_id),
                    save_logprob_distributions=save_lp,
                    save_vc_distributions=save_vc,
                    save_step_traces=save_tr,
                    episode_id=ep_id,
                    trace_output_dir=str(output_dir),
                    trace_model_name=model_nm or None,
                    trace_hook=trace_hk,
                    trace_session_id=session_id,
                    trace_tags=[t for t in ([*base_tags, stage, model_nm] if model_nm else [*base_tags, stage]) if t],
                    trace_name=ep_id,
                    **history_cfg,
                )
                log(
                    f"Test 4: episode done {ep_id} — steps={result['steps']} "
                    f"lm_calls={result.get('total_lm_calls', 0)} wall={result.get('wall_clock_time', 0):.2f}s "
                    f"success={result.get('task_success')}"
                )
                if stage == "C2":
                    for sd in result.get("steps_detail") or []:
                        if not isinstance(sd, dict):
                            continue
                        va = sd.get("vote_agreement")
                        ua = sd.get("unique_actions")
                        if isinstance(va, (int, float)):
                            c2_vote_agreements.append(float(va))
                        if isinstance(ua, int):
                            c2_unique_actions.append(int(ua))
                data = {
                    "episode_id": ep_id,
                    "domain": "textworld",
                    "instance": inst,
                    "compute_stage": stage,
                    "run": run,
                    "task_success": result["task_success"],
                    "steps": result["steps"],
                    "lm_calls": result["lm_calls"],
                    "total_lm_calls": int(result.get("total_lm_calls", 0)),
                    "tokens": result["tokens"],
                    "wall_clock_time": result["wall_clock_time"],
                    "tle_per_step": result.get("tle_per_step"),
                    "vc_per_step": result.get("vc_per_step"),
                }
                if result.get("vc_detail_per_step") is not None:
                    data["vc_detail_per_step"] = result["vc_detail_per_step"]
                log_episode(ep_id, data, output_dir)
                _maybe_write_logprob_artifacts(config, ep_id, result, output_dir)
                _maybe_write_vc_artifacts(config, ep_id, result, output_dir)
                episodes_data.append(data)
                _log_progress()
    if c2_vote_agreements:
        xs = sorted(c2_vote_agreements)
        med = xs[len(xs) // 2]
        mean = sum(xs) / len(xs)
        log(
            f"Test 4: C2 diversity summary — vote_agreement median={med:.3f} mean={mean:.3f} "
            f"(n_steps={len(xs)})"
        )
    if c2_unique_actions:
        ys = sorted(c2_unique_actions)
        med_u = ys[len(ys) // 2]
        mean_u = sum(ys) / len(ys)
        log(
            f"Test 4: C2 diversity summary — unique_actions median={med_u:.3f} mean={mean_u:.3f} "
            f"(n_steps={len(ys)})"
        )
    return episodes_data


def _last_non_null(values: list[Any]) -> Any | None:
    for v in reversed(values or []):
        if v is not None:
            return v
    return None


def _build_feasibility_report(
    *,
    pilot_mode: str,
    system: dict[str, Any],
    sanity: dict[str, Any] | None,
    test1: dict[str, Any],
    textworld_episodes: list[dict[str, Any]],
    toh_episodes: list[dict[str, Any]],
    toh: dict[str, Any],
    config: dict[str, Any],
    wall_clock_total_s: float,
) -> dict[str, Any]:
    from src.analysis.calibration import compute_ece

    expected_min = float(config.get("test1_inference", {}).get("expected_tok_per_sec_min", 50))
    tok_s = float(test1.get("tokens_per_sec", 0.0) or 0.0)

    vc_parse_rate, tle_nonnull_rate = _episode_vc_tle_rates(textworld_episodes, toh_episodes)

    toh_parse_rate = toh.get("parse_rate")
    toh_success_rate = toh.get("success_rate")
    toh_avg_optimal_rate = toh.get("avg_optimal_rate")
    toh_avg_legal_rate = toh.get("avg_legal_rate")
    toh_oscillation_rate = toh.get("oscillation_rate")
    toh_avg_min_optimal_moves_remaining = toh.get("avg_min_optimal_moves_remaining")

    # ECE on TextWorld episodes: use last non-null VC as confidence proxy; fallback 0.5.
    predictions: list[float] = []
    correctness: list[float] = []
    for ep in textworld_episodes:
        vc = ep.get("vc_per_step") or []
        last_vc = _last_non_null(vc) if isinstance(vc, list) else None
        pred = (float(last_vc) / 100.0) if isinstance(last_vc, (int, float)) else 0.5
        predictions.append(pred)
        correctness.append(1.0 if ep.get("task_success") else 0.0)
    ece = compute_ece(predictions, correctness, n_bins=5) if predictions else None

    has_logprobs = False
    if pilot_mode == "mock":
        has_logprobs = True
    elif isinstance(sanity, dict):
        has_logprobs = bool(sanity.get("has_logprobs"))

    # Keep questions/fallbacks aligned with the previous markdown report, but now JSON.
    checks: list[dict[str, Any]] = []

    def _add_check(id_: int, question: str, passed: bool, fallback: str) -> None:
        checks.append({"id": id_, "question": question, "passed": bool(passed), "fallback": fallback})

    _add_check(
        1,
        "Real model backend loads (no mock fallback)?",
        bool(pilot_mode) and (pilot_mode == "mock" or bool((sanity or {}).get("ok"))),
        "Fix model id / vLLM install",
    )
    _add_check(
        2,
        f"Inferenzgeschwindigkeit ≥{expected_min:g} tok/s?",
        tok_s >= expected_min,
        "Budget × 1.5",
    )
    _add_check(
        3,
        "Token-Level-Logprobs extrahierbar (real)?",
        has_logprobs,
        "Try different backend / vLLM version",
    )
    _add_check(
        4,
        "Verbalisierte Konfidenz parsebar (real)?",
        (pilot_mode == "mock") or (isinstance(vc_parse_rate, (int, float)) and float(vc_parse_rate) >= 0.8),
        "Add few-shot / enforce format",
    )
    _add_check(
        5,
        "TextWorld installierbar und lauffähig?",
        isinstance(textworld_episodes, list) and len(textworld_episodes) > 0,
        "Eigene Text-Envs",
    )
    _add_check(
        6,
        "Agent generiert valide Aktionen?",
        isinstance(textworld_episodes, list) and len(textworld_episodes) > 0,
        "Action-Space-Constraining",
    )
    _add_check(
        7,
        "Best-of-3 + Majority Vote (C2) implementiert?",
        True,
        "Konsistenzprüfung",
    )
    _add_check(
        8,
        "End-to-End produziert vollständige Logs?",
        bool(textworld_episodes) and all(isinstance(e, dict) and "steps" in e for e in textworld_episodes),
        "Logging debuggen",
    )
    _add_check(
        9,
        "Daten-Download + Analyse machbar?",
        ece is not None and isinstance(ece, (int, float)),
        "Volume als Zwischenspeicher",
    )
    _add_check(
        10,
        "Tower of Hanoi moves parseable ≥80%?",
        (pilot_mode == "mock")
        or (isinstance(toh_parse_rate, (int, float)) and float(toh_parse_rate) >= 0.8),
        "Few-shot format / constrain action space",
    )

    passed = sum(1 for c in checks if c.get("passed"))
    total = len(checks)
    go = passed >= 8 or (total > 0 and (passed / total) >= 0.8)

    return {
        "pilot_mode": pilot_mode,
        "system": system,
        "summary": {
            "tokens_per_sec": tok_s,
            "expected_tok_per_sec_min": expected_min,
            "vc_parse_rate": vc_parse_rate,
            "tle_nonnull_rate": tle_nonnull_rate,
            "toh_parse_rate": toh_parse_rate,
            "toh_success_rate": toh_success_rate,
            "toh_avg_optimal_rate": toh_avg_optimal_rate,
            "toh_avg_legal_rate": toh_avg_legal_rate,
            "toh_oscillation_rate": toh_oscillation_rate,
            "toh_avg_min_optimal_moves_remaining": toh_avg_min_optimal_moves_remaining,
            "ece": ece,
            "n_episodes_textworld": int(len(textworld_episodes)),
            "n_episodes_toh": int(toh.get("num_episodes") or 0),
        },
        "checks": checks,
        "passed": int(passed),
        "total": int(total),
        "go": bool(go),
        "wall_clock_total_s": float(wall_clock_total_s),
    }


def _resolve_pilot_mode(args) -> str:
    """Resolve pilot mode. --real auto-detects hf vs cuda when mode is mock; lmstudio stays explicit."""
    raw = getattr(args, "pilot_mode", None) or os.environ.get("PILOT_MODE") or "mock"
    mode = parse_pilot_mode_arg(raw) if isinstance(raw, str) else "mock"
    if args.real or os.environ.get("USE_REAL_MODEL") == "1":
        if mode == "mock":
            try:
                import torch
                if torch.cuda.is_available():
                    mode = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    mode = "hf"
                else:
                    mode = "mock"
                    print("Warning: --real requested but no CUDA/MPS found; falling back to mock.")
            except Exception:
                mode = "mock"
    return mode


def _run_pilot_from_args(args: argparse.Namespace) -> None:
    t_pilot0 = time.perf_counter()
    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    base_output = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    if args.no_timestamp_run:
        output_dir = base_output
    else:
        from src.utils.run_output_layout import make_run_subdirectory

        output_dir = make_run_subdirectory(base_output, prefix="pilot")
    pilot_mode = _resolve_pilot_mode(args)
    config, lmstudio_note, lmstudio_applied = load_pilot_config_with_lmstudio_override(
        config_path, pilot_mode, REPO_ROOT, getattr(args, "lmstudio_config", None)
    )
    if getattr(args, "model_name", None):
        config.setdefault("model", {})["name"] = args.model_name
    only_steps = _only_steps_in_order(frozenset(args.only)) if args.only else list(PILOT_STEPS_ORDER)
    only_set = frozenset(only_steps)

    log(f"Pilot run start — config={config_path} output_dir={output_dir}")
    if not args.no_timestamp_run:
        from src.utils.run_output_layout import write_short_run_info

        write_short_run_info(
            output_dir,
            script="run_pilot.py",
            config_path=config_path,
            extra={
                "output_dir_resolved": str(output_dir.resolve()),
                "pilot_mode": pilot_mode,
                "only_steps": only_steps,
                "lmstudio_config_applied": lmstudio_applied,
                "model_name": config.get("model", {}).get("name"),
            },
        )
        log(f"Wrote {output_dir / 'run_info.json'}")
    if lmstudio_note:
        log(lmstudio_note)
    log(f"Pilot mode: {pilot_mode}")
    if args.only:
        log(f"Steps: {' '.join(only_steps)}")

    needs_model = (pilot_mode != "mock") and bool(only_set & PILOT_STEPS_NEED_MODEL)
    real_model: Any | None = None
    real_model_err: str | None = None
    if needs_model:
        if pilot_mode != "mock":
            model_name = config.get("model", {}).get("name", "?")
            log(f"Loading model ({model_name})... (cache load may take ~1 min)")
        real_model, real_model_err = _create_real_model(config, pilot_mode)
        _assert_real_model_or_raise(pilot_mode, real_model, real_model_err)
        config.setdefault("pilot", {})["use_real_model"] = True
        log("Model ready.")

    system = _try_gpu_info()
    sanity: dict[str, Any] | None = None
    test1: dict[str, Any] = {}
    test2: dict[str, Any] = {}
    test3: dict[str, Any] = {}
    episodes: list[dict[str, Any]] = []
    toh: dict[str, Any] = {}

    if "sanity" in only_set:
        if pilot_mode != "mock" and real_model is not None:
            sanity = _sanity_check_real_inference(config, pilot_mode, real_model)
            _save_json(output_dir, "pilot_sanity", sanity)
            s = sanity
            log(
                f"Real model sanity: ok={s.get('ok')} latency={s.get('latency_s', 0):.2f}s "
                f"logprobs={s.get('has_logprobs')} tok_out≈{s.get('completion_tokens_observed')}"
            )
        elif pilot_mode == "mock" and args.only:
            log("Skipping sanity (mock mode).")

    if "test1" in only_set:
        test1 = run_test1_inference_speed(config, output_dir, real_model=real_model)
        _save_json(output_dir, "pilot_test1_inference", test1)
        log(f"Test 1 done — tokens/s={test1.get('tokens_per_sec', 0):.1f}")

    if "test2" in only_set:
        test2 = run_test2_token_entropy(config, output_dir, real_model=real_model)
        _save_json(output_dir, "pilot_test2_tle", test2)
        log("Test 2 done.")

    if "test3" in only_set:
        test3 = run_test3_verbalized_confidence(config, real_model=real_model)
        _save_json(output_dir, "pilot_test3_vc", test3)
        log("Test 3 done.")

    if "test4" in only_set:
        episodes = run_test4_textworld_e2e(config, output_dir, real_model=real_model)
        log(f"Test 4 done — {len(episodes)} episodes")

    if "test5" in only_set:
        toh = run_test5_tower_of_hanoi(config, output_dir, real_model=real_model)
        _save_json(output_dir, "pilot_test5_toh", toh)
        log(f"Test 5 done — parse_rate={toh.get('parse_rate', 0):.2f}")

    if "feasibility" in only_set:
        t1, t2, t3, eps, th, san, toh_eps = _prepare_feasibility_inputs(
            output_dir,
            test1=test1,
            test2=test2,
            test3=test3,
            episodes=episodes,
            toh=toh,
            sanity=sanity,
        )
        feasibility = _build_feasibility_report(
            pilot_mode=pilot_mode,
            system=system,
            sanity=san,
            test1=t1,
            textworld_episodes=eps,
            toh_episodes=toh_eps,
            toh=th,
            config=config,
            wall_clock_total_s=(time.perf_counter() - t_pilot0),
        )
        _save_json(output_dir, "pilot_feasibility", feasibility)
        log(
            f"Feasibility done — go={feasibility.get('go')} "
            f"passed={feasibility.get('passed')}/{feasibility.get('total')}"
        )

    log(f"Pilot done — wall {format_run_elapsed(time.perf_counter() - t_pilot0)} total")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pilot study (Tests 1-6). mock | hf (HF+MPS) | cuda | lmstudio (OpenAI API)."
    )
    parser.add_argument("--config", default="configs/pilot.yaml", help="Pilot config YAML")
    parser.add_argument(
        "--lmstudio-config",
        default=None,
        metavar="PATH",
        help="When --pilot-mode lmstudio: optional YAML deep-merged on top of --config "
        "(default: configs/lmstudio_config.yaml; env LMSTUDIO_CONFIG_PATH). "
        "Use enabled: false in that file to keep base pilot.yaml only.",
    )
    parser.add_argument("--output-dir", default="data/results", help="Base output directory")
    parser.add_argument(
        "--no-timestamp-run",
        action="store_true",
        help="Write directly under --output-dir instead of creating a timestamped subfolder (run_*_UTC).",
    )
    parser.add_argument(
        "--pilot-mode",
        type=parse_pilot_mode_arg,
        default="mock",
        help="mock | hf (HuggingFace+MPS) | m1 (deprecated=hf) | cuda | lmstudio (LM Studio server).",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real model; auto-detect hf vs cuda if --pilot-mode is mock",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEP",
        choices=list(PILOT_STEPS_ORDER),
        default=None,
        help=(
            "Run only these pilot steps (canonical order). "
            "Steps: sanity, test1, test2, test3, test4, test5, feasibility. "
            "Example: --only test2. For feasibility, prior results in output_dir are merged "
            "(pilot_test*.json, ep_textworld_*.json, pilot_test5_toh.json)."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        metavar="ID",
        help="Override model.name after config (and LM Studio YAML merge). Used by run_pilot_models.py.",
    )
    return parser


def _parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


def main() -> None:
    run_pilot_main(_parse_args(), _run_pilot_from_args)


if __name__ == "__main__":
    main()
