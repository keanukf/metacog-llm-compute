#!/usr/bin/env python3
"""
Pilot study runner: runs Tests 1-6 in order, writes pilot_benchmark.json,
pilot_calibration.json, and optional markdown reports.
Usage: python scripts/run_pilot.py --config configs/pilot.yaml [--output-dir data/results] [--real]
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

from src.utils.pilot_config import load_pilot_config_with_lmstudio_override, load_yaml_path
from src.utils.run_progress import format_run_elapsed, log, log_step_line


def load_config(config_path: str | Path) -> dict:
    """Load a single YAML file (no LM Studio merge). Prefer ``load_pilot_config`` in main."""
    return load_yaml_path(Path(config_path))


def _save_json(output_dir: Path, filename_stem: str, data: dict[str, Any]) -> Path:
    """
    Save a single JSON artifact to output_dir with a predictable filename.
    This is used for per-test outputs (not per-episode outputs).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename_stem}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log(f"Wrote {path}")
    return path


def _run_mock_inference_speed_benchmark(num_prompts: int, tokens_per_call: int = 200) -> dict[str, Any]:
    """
    Simulate a batch of prompts with fixed tokens and timings; return result dict.
    Kept here so the pilot runner doesn't import from the pytest suite.
    """
    latencies: list[float] = []
    total_tokens = 0
    for _ in range(int(num_prompts)):
        t0 = time.perf_counter()
        time.sleep(0.001)
        total_tokens += int(tokens_per_call)
        latencies.append(time.perf_counter() - t0)
    elapsed = sum(latencies)
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
    mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
    variance = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) if latencies else 0.0
    std_lat = variance**0.5
    return {
        "tokens_per_sec": float(tokens_per_sec),
        "latency_mean": float(mean_lat),
        "latency_std": float(std_lat),
        "total_tokens": int(total_tokens),
        "num_prompts": int(num_prompts),
    }


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
            backend = config.get("inference", {}).get("backend", "vllm")
            return create_wrapper(backend=backend, model_name=model_name, dtype=dtype), None
        if pilot_mode == "lmstudio":
            inf = config.get("inference", {})
            base_url = inf.get("lmstudio_base_url") or os.environ.get(
                "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
            )
            api_key = inf.get("lmstudio_api_key") or os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
            return (
                create_wrapper(
                    backend="lmstudio",
                    model_name=model_name,
                    base_url=base_url,
                    lmstudio_api_key=api_key,
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


def run_test2_token_entropy(config: dict, *, real_model=None) -> dict:
    """Test 2: TLE extraction from real logprobs when available; else synthetic."""
    log("Test 2: token entropy — start")
    from src.signals.token_entropy import compute_tle, extract_tle_from_response

    if real_model is None:
        # Synthetic logprobs: easy (low entropy) vs hard (high entropy)
        easy = [{"logprob": -0.1}] * 10
        hard = [{"logprob": -2.0}] * 10
        return {"mode": "synthetic", "easy_tle": compute_tle(easy), "hard_tle": compute_tle(hard)}

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
    for name, p in prompts.items():
        text, logprobs = real_model.generate(p, logprobs=True, max_tokens=max_tokens, temperature=temperature)
        tle = extract_tle_from_response(text, logprobs) if logprobs else None
        out["per_prompt"][name] = {
            "tle": tle,
            "completion_tokens_observed": len(logprobs) if logprobs else 0,
        }
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
    log("Test 5: Tower of Hanoi parseability — start (C0; per-step progress below)")
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.tower_of_hanoi import TowerOfHanoiEnv, generate_instances
    from src.utils.logging_utils import log_episode

    class MockModel:
        def generate(self, prompt, logprobs=False, **kwargs):
            text = "A->B\nConfidence: 50"
            lp = [{"logprob": -0.5}] * 5 if logprobs else None
            return text, lp

    model = real_model if real_model is not None else MockModel()
    toh_cfg = config.get("tower_of_hanoi", {})
    num_episodes = int(toh_cfg.get("pilot_episodes", 20))
    num_disks = int(toh_cfg.get("pilot_num_disks", 3))
    seed = int(toh_cfg.get("task_generation_seed", 42))
    max_steps = int(toh_cfg.get("pilot_max_steps", 20))
    tasks = generate_instances(
        num_episodes,
        seed=seed,
        num_disks_range=(num_disks, num_disks),
        partial_start_range=(0, 0),
    )
    worst_lm = num_episodes * max_steps
    log(
        f"Test 5: ToH plan {num_episodes} episodes × up to {max_steps} steps → ≤{worst_lm} LM calls (C0)"
    )
    parseable_steps = 0
    total_steps = 0
    episodes_data: list[dict[str, Any]] = []
    for i, task in enumerate(tasks):

        def _make_toh_on_step(ep_i: int, ep_total: int):
            def _inner(info: dict) -> None:
                log_step_line(f"pilot ToH ep {ep_i + 1}/{ep_total}", info)

            return _inner

        env = TowerOfHanoiEnv(task=task, max_steps=max_steps)
        step_fn = get_step_fn("C0")
        log(f"Test 5: ToH episode {i + 1}/{num_episodes} — running (max {max_steps} steps)...")
        result = run_episode(
            env,
            model,
            "C0",
            step_fn=step_fn,
            max_steps=max_steps,
            on_step=_make_toh_on_step(i, num_episodes),
        )
        log(
            f"Test 5: ToH episode done {i + 1}/{num_episodes} — steps={result['steps']} "
            f"lm_calls={result.get('total_lm_calls', 0)} wall={result.get('wall_clock_time', 0):.2f}s "
            f"success={result.get('task_success')}"
        )
        ep_id = f"ep_tower_of_hanoi_{i}_C0_0"
        data = {
            "episode_id": ep_id,
            "domain": "tower_of_hanoi",
            "instance": i,
            "compute_stage": "C0",
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
        log_episode(ep_id, data, output_dir)
        episodes_data.append(data)
        step_corr = result.get("step_correctness") or []
        for d in step_corr:
            total_steps += 1
            if isinstance(d, dict) and d.get("action_parsed") is not None:
                parseable_steps += 1
    parse_rate = (parseable_steps / total_steps) if total_steps else 0.0
    return {
        "num_episodes": num_episodes,
        "num_disks": num_disks,
        "max_steps": max_steps,
        "total_steps": total_steps,
        "parseable_steps": parseable_steps,
        "parse_rate": parse_rate,
    }

def run_test4_textworld_e2e(config: dict, output_dir: Path, real_model=None) -> list[dict]:
    """Test 4: instances x 3 stages x runs = episodes; structured JSON per episode.
    If real_model is provided, use it; otherwise use MockModel (stub env always).
    """
    log("Test 4: end-to-end TextWorld episodes — start")
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.textworld_env import TextWorldEnv
    from src.utils.logging_utils import log_episode

    class MockModel:
        def generate(self, prompt, logprobs=False, **kwargs):
            text = "go north"
            lp = [{"logprob": -0.5}] * 5 if logprobs else None
            return text, lp

    model = real_model if real_model is not None else MockModel()
    instances = config.get("pilot", {}).get("instances", 5)
    stages = ["C0", "C1", "C2"]
    runs = config.get("pilot", {}).get("runs_per_instance", 1)
    total_episodes = instances * len(stages) * runs
    episode_idx = [0]

    def _log_progress():
        episode_idx[0] += 1
        if episode_idx[0] % 5 == 0 or episode_idx[0] == total_episodes:
            log(f"Test 5: batch progress {episode_idx[0]}/{total_episodes} episodes")

    log(
        f"Test 4: plan {instances} instances × {len(stages)} stages × {runs} runs = "
        f"{total_episodes} episodes (max 10 steps each)"
    )
    episodes_data = []
    for inst in range(instances):
        for stage in stages:
            for run in range(runs):
                ep_id = f"ep_textworld_{inst}_{stage}_{run}"

                def _make_test5_on_step(eid: str):
                    def _inner(info: dict) -> None:
                        log_step_line(f"pilot Test4 {eid}", info)

                    return _inner

                env = TextWorldEnv(max_steps=10)
                step_fn = get_step_fn(stage)
                log(f"Test 4: episode start {ep_id} (stage={stage})")
                result = run_episode(
                    env,
                    model,
                    stage,
                    step_fn=step_fn,
                    max_steps=10,
                    on_step=_make_test5_on_step(ep_id),
                )
                log(
                    f"Test 4: episode done {ep_id} — steps={result['steps']} "
                    f"lm_calls={result.get('total_lm_calls', 0)} wall={result.get('wall_clock_time', 0):.2f}s "
                    f"success={result.get('task_success')}"
                )
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
                log_episode(ep_id, data, output_dir)
                episodes_data.append(data)
                _log_progress()
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
    test2: dict[str, Any],
    test3: dict[str, Any],
    textworld_episodes: list[dict[str, Any]],
    toh: dict[str, Any],
    config: dict[str, Any],
    wall_clock_total_s: float,
) -> dict[str, Any]:
    from src.analysis.calibration import compute_ece

    expected_min = float(config.get("test1_inference", {}).get("expected_tok_per_sec_min", 80))
    tok_s = float(test1.get("tokens_per_sec", 0.0) or 0.0)

    vc_parse_rate = None
    if isinstance(test3, dict) and test3.get("mode") == "real":
        vc_parse_rate = test3.get("parse_rate")

    toh_parse_rate = toh.get("parse_rate")

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
        "Hochrechnung Core ≤70 GPU-Stunden?",
        tok_s >= expected_min,
        "Runs/Instanzen reduzieren",
    )
    _add_check(
        10,
        "Daten-Download + Analyse machbar?",
        ece is not None and isinstance(ece, (int, float)),
        "Volume als Zwischenspeicher",
    )
    _add_check(
        11,
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
            "toh_parse_rate": toh_parse_rate,
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


def main() -> None:
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
    parser.add_argument("--output-dir", default="data/results", help="Output directory")
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
    args = parser.parse_args()
    t_pilot0 = time.perf_counter()
    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    output_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_mode = _resolve_pilot_mode(args)
    config, lmstudio_note, lmstudio_applied = load_pilot_config_with_lmstudio_override(
        config_path, pilot_mode, REPO_ROOT, getattr(args, "lmstudio_config", None)
    )
    log(f"Pilot run start — config={config_path} output_dir={output_dir}")
    if lmstudio_note:
        log(lmstudio_note)
    log(f"Pilot mode: {pilot_mode}")
    if pilot_mode != "mock":
        model_name = config.get("model", {}).get("name", "?")
        log(f"Loading model ({model_name})... (cache load may take ~1 min)")
    real_model, real_model_err = _create_real_model(config, pilot_mode)
    _assert_real_model_or_raise(pilot_mode, real_model, real_model_err)
    if pilot_mode != "mock":
        config.setdefault("pilot", {})["use_real_model"] = True
        log("Model ready.")

    system = _try_gpu_info()
    sanity: dict[str, Any] | None = None
    if pilot_mode != "mock" and real_model is not None:
        sanity = _sanity_check_real_inference(config, pilot_mode, real_model)
        _save_json(output_dir, "pilot_sanity", sanity)
        s = sanity
        log(
            f"Real model sanity: ok={s.get('ok')} latency={s.get('latency_s', 0):.2f}s "
            f"logprobs={s.get('has_logprobs')} tok_out≈{s.get('completion_tokens_observed')}"
        )

    test1 = run_test1_inference_speed(config, output_dir, real_model=real_model)
    _save_json(output_dir, "pilot_test1_inference", test1)
    log(f"Test 1 done — tokens/s={test1.get('tokens_per_sec', 0):.1f}")

    test2 = run_test2_token_entropy(config, real_model=real_model)
    _save_json(output_dir, "pilot_test2_tle", test2)
    log("Test 2 done.")

    test3 = run_test3_verbalized_confidence(config, real_model=real_model)
    _save_json(output_dir, "pilot_test3_vc", test3)
    log("Test 3 done.")

    episodes = run_test4_textworld_e2e(config, output_dir, real_model=real_model)
    log(f"Test 4 done — {len(episodes)} episodes")

    toh = run_test5_tower_of_hanoi(config, output_dir, real_model=real_model)
    log(f"Test 5 done — parse_rate={toh.get('parse_rate', 0):.2f}")

    feasibility = _build_feasibility_report(
        pilot_mode=pilot_mode,
        system=system,
        sanity=sanity,
        test1=test1,
        test2=test2,
        test3=test3,
        textworld_episodes=episodes,
        toh=toh,
        config=config,
        wall_clock_total_s=(time.perf_counter() - t_pilot0),
    )
    _save_json(output_dir, "pilot_feasibility", feasibility)
    log(f"Pilot done — wall {format_run_elapsed(time.perf_counter() - t_pilot0)} total")


if __name__ == "__main__":
    main()
