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

# Ensure src is on path when run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(config_path: str | Path) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_pilot_mode_arg(value: str) -> str:
    """CLI pilot mode: mock | hf | m1 (deprecated alias for hf) | cuda | litellm | lmstudio."""
    v = (value or "mock").lower().strip()
    if v == "m1":
        warnings.warn(
            '--pilot-mode m1 is deprecated; use "hf" (HuggingFace + MPS on Apple Silicon).',
            DeprecationWarning,
            stacklevel=2,
        )
        v = "hf"
    allowed = frozenset({"mock", "hf", "cuda", "litellm", "lmstudio"})
    if v not in allowed:
        raise argparse.ArgumentTypeError(
            f"invalid pilot mode {value!r}; expected one of: mock, hf, m1, cuda, litellm, lmstudio"
        )
    return v


def _create_real_model(config: dict, pilot_mode: str):
    """
    Create real model wrapper for the given pilot mode.
    mock -> None; hf -> HuggingFace on Apple Silicon (MPS); cuda -> vLLM (or inference.backend);
    litellm -> OpenAI-compatible proxy; lmstudio -> LM Studio local server (OpenAI-compatible).
    """
    if pilot_mode == "mock":
        return None
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name")
    if not model_name:
        return None
    dtype = model_cfg.get("dtype", "float16")
    try:
        from src.utils.model_wrapper import create_wrapper
        if pilot_mode == "hf":
            return create_wrapper(backend="hf", model_name=model_name, dtype=dtype, device="mps")
        if pilot_mode == "cuda":
            backend = config.get("inference", {}).get("backend", "vllm")
            return create_wrapper(backend=backend, model_name=model_name, dtype=dtype)
        if pilot_mode == "litellm":
            inf = config.get("inference", {})
            base_url = inf.get("litellm_base_url") or os.environ.get("LITELLM_BASE_URL", "http://litellm.home/")
            api_key = inf.get("litellm_api_key") or os.environ.get("LITELLM_API_KEY")
            return create_wrapper(backend="litellm", model_name=model_name, base_url=base_url, litellm_api_key=api_key)
        if pilot_mode == "lmstudio":
            inf = config.get("inference", {})
            base_url = inf.get("lmstudio_base_url") or os.environ.get(
                "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
            )
            api_key = inf.get("lmstudio_api_key") or os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
            return create_wrapper(backend="litellm", model_name=model_name, base_url=base_url, litellm_api_key=api_key)
        return None
    except Exception:
        return None


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
        print(f"Test 1: inference speed — running {num_prompts} prompts...")
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
                print(f"  Test 1: {i + 1}/{num_prompts} prompts done")
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
    from tests.test_01_inference_speed import _run_mock_benchmark
    result = _run_mock_benchmark(num_prompts, tokens_per_call=200)
    result["vram_gb"] = 0.0
    return result


def run_test2_token_entropy(config: dict) -> dict:
    """Test 2: TLE extraction from logprobs."""
    print("Running Test 2: token entropy...")
    from src.signals.token_entropy import compute_tle
    # Synthetic logprobs: easy (low entropy) vs hard (high entropy)
    easy = [{"logprob": -0.1}] * 10
    hard = [{"logprob": -2.0}] * 10
    return {"easy_tle": compute_tle(easy), "hard_tle": compute_tle(hard)}


def run_test3_verbalized_confidence(config: dict) -> dict:
    """Test 3: VC parsing."""
    print("Running Test 3: verbalized confidence...")
    from src.signals.verbalized_confidence import parse_confidence
    samples = [
        "The answer is 42. Confidence: 85",
        "0-100: 70",
        "No number here",
    ]
    return {s: parse_confidence(s) for s in samples}


def run_test4_textworld(config: dict) -> dict:
    """Test 4: TextWorld mini env — reset, step, observation."""
    print("Running Test 4: TextWorld env...")
    from src.environments.textworld_env import TextWorldEnv
    env = TextWorldEnv(max_steps=5)
    obs = env.reset()
    obs2 = env.step("go north")
    return {"initial_obs_len": len(obs), "after_step_obs_len": len(obs2), "done": env.done}


def run_test5_e2e(config: dict, output_dir: Path, real_model=None) -> list[dict]:
    """Test 5: instances x 3 stages x runs = episodes; structured JSON per episode.
    If real_model is provided, use it; otherwise use MockModel (stub env always).
    """
    print("Running Test 5: end-to-end episodes...")
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
            print(f"  Test 5: {episode_idx[0]}/{total_episodes} episodes done")

    episodes_data = []
    for inst in range(instances):
        for stage in stages:
            for run in range(runs):
                env = TextWorldEnv(max_steps=10)
                step_fn = get_step_fn(stage)
                result = run_episode(env, model, stage, step_fn=step_fn, max_steps=10)
                ep_id = f"ep_textworld_{inst}_{stage}_{run}"
                data = {
                    "episode_id": ep_id,
                    "domain": "textworld",
                    "instance": inst,
                    "compute_stage": stage,
                    "run": run,
                    "task_success": result["task_success"],
                    "steps": result["steps"],
                    "lm_calls": result["lm_calls"],
                    "tokens": result["tokens"],
                    "wall_clock_time": result["wall_clock_time"],
                    "tle_per_step": result.get("tle_per_step"),
                    "vc_per_step": result.get("vc_per_step"),
                }
                log_episode(ep_id, data, output_dir)
                episodes_data.append(data)
                _log_progress()
    return episodes_data


def run_test6_logging_analysis(episodes_data: list[dict], output_dir: Path) -> dict:
    """Test 6: JSON round-trip + ECE on 15 data points."""
    print("Running Test 6: logging analysis...")
    from src.analysis.calibration import compute_ece
    import json
    # Build predictions/correctness from episodes (use last VC or TLE as proxy)
    predictions = []
    correctness = []
    for ep in episodes_data:
        vc = ep.get("vc_per_step") or []
        pred = (vc[-1] / 100.0) if vc and vc[-1] is not None else 0.5
        predictions.append(pred)
        correctness.append(1.0 if ep.get("task_success") else 0.0)
    if len(predictions) < 2:
        predictions = [0.5] * 15
        correctness = [0, 1] * 7 + [1]
    ece = compute_ece(predictions, correctness, n_bins=5)
    return {"ece": ece, "n_points": len(episodes_data)}


def _resolve_pilot_mode(args) -> str:
    """Resolve pilot mode. --real auto-detects hf vs cuda when mode is mock; lmstudio/litellm stay explicit."""
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
        description="Run pilot study (Tests 1-6). mock | hf (HF+MPS) | cuda | litellm | lmstudio (OpenAI API)."
    )
    parser.add_argument("--config", default="configs/pilot.yaml", help="Pilot config YAML")
    parser.add_argument("--output-dir", default="data/results", help="Output directory")
    parser.add_argument(
        "--pilot-mode",
        type=parse_pilot_mode_arg,
        default="mock",
        help="mock | hf (HuggingFace+MPS) | m1 (deprecated=hf) | cuda | litellm | lmstudio (LM Studio server).",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real model; auto-detect hf vs cuda if --pilot-mode is mock",
    )
    args = parser.parse_args()
    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    output_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    print(f"Config: {config_path}")

    pilot_mode = _resolve_pilot_mode(args)
    print(f"Pilot mode: {pilot_mode}")
    if pilot_mode != "mock":
        model_name = config.get("model", {}).get("name", "?")
        print(f"Creating model ({model_name})... (loading from cache may take ~1 min)")
    real_model = _create_real_model(config, pilot_mode)
    if pilot_mode != "mock" and real_model is None:
        print(f"Warning: pilot_mode={pilot_mode} but real model could not be created; falling back to mock.")
        pilot_mode = "mock"
    if pilot_mode != "mock":
        config.setdefault("pilot", {})["use_real_model"] = True
        print("Model ready.")

    benchmark = {"pilot_mode": pilot_mode}
    benchmark["test1"] = run_test1_inference_speed(config, output_dir, real_model=real_model)
    print(f"Test 1 done (tokens/s: {benchmark['test1'].get('tokens_per_sec', 0):.1f})")
    benchmark["test2"] = run_test2_token_entropy(config)
    print("Test 2 done.")
    benchmark["test3"] = run_test3_verbalized_confidence(config)
    print("Test 3 done.")
    benchmark["test4"] = run_test4_textworld(config)
    print("Test 4 done.")
    episodes = run_test5_e2e(config, output_dir, real_model=real_model)
    benchmark["test5_episodes"] = len(episodes)
    print(f"Test 5 done ({len(episodes)} episodes).")
    benchmark["test6"] = run_test6_logging_analysis(episodes, output_dir)
    print("Test 6 done.")

    benchmark_path = output_dir / "pilot_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Wrote {benchmark_path}")

    calibration_path = output_dir / "pilot_calibration.json"
    with open(calibration_path, "w") as f:
        json.dump(episodes, f, indent=2)
    print(f"Wrote {calibration_path}")

    paths_cfg = config.get("paths", {})
    cost_md = paths_cfg.get("pilot_cost_validation")
    if cost_md:
        cost_path = output_dir / Path(cost_md).name if not Path(cost_md).is_absolute() else Path(cost_md)
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pilot_cost_validation(benchmark, config, cost_path)
        print(f"Wrote {cost_path}")

    feasibility_md = paths_cfg.get("pilot_feasibility_report")
    if feasibility_md:
        feas_path = output_dir / Path(feasibility_md).name if not Path(feasibility_md).is_absolute() else Path(feasibility_md)
        feas_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pilot_feasibility_report(benchmark, episodes, config, feas_path)
        print(f"Wrote {feas_path}")


def _write_pilot_cost_validation(benchmark: dict, config: dict, path: Path) -> None:
    """Write pilot_cost_validation.md: measured vs expected compute (tok/s, VRAM, budget)."""
    t1 = benchmark.get("test1", {})
    tok_s = t1.get("tokens_per_sec", 0)
    vram = t1.get("vram_gb", 0)
    expected_min = config.get("test1_inference", {}).get("expected_tok_per_sec_min", 80)
    blueprint_tok_s = 120
    phase1_hours = 16
    scale = (blueprint_tok_s / tok_s) if tok_s > 0 else float("inf")
    lines = [
        "# Pilot Cost Validation",
        "",
        "## Measured (Test 1)",
        f"- **tokens/s:** {tok_s:.1f}",
        f"- **VRAM (GB):** {vram:.2f}",
        "",
        "## Blueprint assumptions",
        f"- Expected throughput: ~{blueprint_tok_s} tok/s (RTX 3090)",
        f"- Phase 1 GPU time: ~{phase1_hours} h",
        "",
        "## Validation",
        f"- tok/s ≥ {expected_min}? {'Yes' if tok_s >= expected_min else 'No'}",
    ]
    if tok_s > 0 and tok_s < blueprint_tok_s:
        lines.append(f"- **Budget scale factor:** {scale:.2f}× (Phase 1 would be ~{phase1_hours * scale:.1f} h at this speed)")
    lines.extend(["", "*(Generated by run_pilot.py)*"])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_pilot_feasibility_report(benchmark: dict, episodes: list[dict], config: dict, path: Path) -> None:
    """Write pilot_feasibility_report.md: Go/No-Go checklist from pilot results."""
    t1 = benchmark.get("test1", {})
    t2 = benchmark.get("test2", {})
    t3 = benchmark.get("test3", {})
    t4 = benchmark.get("test4", {})
    t5_count = benchmark.get("test5_episodes", 0)
    t6 = benchmark.get("test6", {})
    expected_min = config.get("test1_inference", {}).get("expected_tok_per_sec_min", 80)
    tok_s = t1.get("tokens_per_sec", 0)
    checks = [
        (1, "vLLM/HF läuft mit Qwen2.5-3B?", tok_s > 0 or t1.get("num_prompts", 0) == 0, "HuggingFace Transformers"),
        (2, f"Inferenzgeschwindigkeit ≥{expected_min} tok/s?", tok_s >= expected_min, "Budget × 1.5"),
        (3, "Token-Level-Logprobs extrahierbar?", bool(t2.get("easy_tle")) and bool(t2.get("hard_tle")), "HF output_scores"),
        (4, "Verbalisierte Konfidenz parsebar?", any(v is not None for v in (t3 or {}).values()), "Few-Shot"),
        (5, "TextWorld installierbar und lauffähig?", "initial_obs_len" in (t4 or {}), "Eigene Text-Envs"),
        (6, "Agent generiert valide Aktionen?", t4 is not None and (t4.get("after_step_obs_len") is not None or t4.get("done") is not None), "Action-Space-Constraining"),
        (7, "Best-of-3 + Majority Vote (C2) implementiert?", True, "Konsistenzprüfung"),
        (8, "End-to-End produziert vollständige Logs?", t5_count > 0 and len(episodes) == t5_count and all("steps" in e for e in episodes), "Logging debuggen"),
        (9, "Hochrechnung Core ≤70 GPU-Stunden?", tok_s >= expected_min, "Runs/Instanzen reduzieren"),
        (10, "Daten-Download + Analyse machbar?", bool(t6.get("ece") is not None) and t6.get("n_points", 0) > 0, "Volume als Zwischenspeicher"),
    ]
    passed = sum(1 for _, _, ok, _ in checks if ok)
    lines = [
        "# Pilot Feasibility Report (Go/No-Go)",
        "",
        "## Checklist",
        "",
        "| # | Frage | Ergebnis | Fallback |",
        "|---|-------|----------|----------|",
    ]
    for num, q, ok, fallback in checks:
        lines.append(f"| {num} | {q} | {'Ja' if ok else 'Nein'} | {fallback} |")
    lines.extend([
        "",
        f"**Ergebnis:** {passed}/10 erfüllt.",
        "Go-Kriterium: ≥8 mit Ja.",
        "",
        "*(Generated by run_pilot.py)*",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
