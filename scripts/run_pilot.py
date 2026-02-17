#!/usr/bin/env python3
"""
Pilot study runner: runs Tests 1-6 in order, writes pilot_benchmark.json,
pilot_calibration.json, and optional markdown reports.
Usage: python scripts/run_pilot.py --config configs/pilot.yaml [--output-dir data/results]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is on path when run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(config_path: str | Path) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_test1_inference_speed(config: dict, output_dir: Path) -> dict:
    """Test 1: Inferenzgeschwindigkeit — 50 prompts, tok/s, latency, VRAM."""
    # Stub: no real model; return structure expected by pilot_benchmark
    return {
        "tokens_per_sec": 0.0,
        "latency_mean": 0.0,
        "latency_std": 0.0,
        "vram_gb": 0.0,
        "num_prompts": config.get("test1_inference", {}).get("num_prompts", 50),
    }


def run_test2_token_entropy(config: dict) -> dict:
    """Test 2: TLE extraction from logprobs."""
    from src.signals.token_entropy import compute_tle
    # Synthetic logprobs: easy (low entropy) vs hard (high entropy)
    easy = [{"logprob": -0.1}] * 10
    hard = [{"logprob": -2.0}] * 10
    return {"easy_tle": compute_tle(easy), "hard_tle": compute_tle(hard)}


def run_test3_verbalized_confidence(config: dict) -> dict:
    """Test 3: VC parsing."""
    from src.signals.verbalized_confidence import parse_confidence
    samples = [
        "The answer is 42. Confidence: 85",
        "0-100: 70",
        "No number here",
    ]
    return {s: parse_confidence(s) for s in samples}


def run_test4_textworld(config: dict) -> dict:
    """Test 4: TextWorld mini env — reset, step, observation."""
    from src.environments.textworld_env import TextWorldEnv
    env = TextWorldEnv(max_steps=5)
    obs = env.reset()
    obs2 = env.step("go north")
    return {"initial_obs_len": len(obs), "after_step_obs_len": len(obs2), "done": env.done}


def run_test5_e2e(config: dict, output_dir: Path) -> list[dict]:
    """Test 5: 5 instances x 3 stages x 1 run = 15 episodes; structured JSON per episode."""
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    from src.environments.textworld_env import TextWorldEnv
    from src.utils.logging_utils import log_episode

    class MockModel:
        def generate(self, prompt, logprobs=False, **kwargs):
            text = "go north"
            lp = [{"logprob": -0.5}] * 5 if logprobs else None
            return text, lp

    instances = config.get("pilot", {}).get("instances", 5)
    stages = ["C0", "C1", "C2"]
    runs = config.get("pilot", {}).get("runs_per_instance", 1)
    episodes_data = []
    for inst in range(instances):
        for stage in stages:
            for run in range(runs):
                env = TextWorldEnv(max_steps=10)
                model = MockModel()
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
    return episodes_data


def run_test6_logging_analysis(episodes_data: list[dict], output_dir: Path) -> dict:
    """Test 6: JSON round-trip + ECE on 15 data points."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pilot study (Tests 1-6)")
    parser.add_argument("--config", default="configs/pilot.yaml", help="Pilot config YAML")
    parser.add_argument("--output-dir", default="data/results", help="Output directory")
    args = parser.parse_args()
    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    output_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)

    benchmark = {}
    benchmark["test1"] = run_test1_inference_speed(config, output_dir)
    benchmark["test2"] = run_test2_token_entropy(config)
    benchmark["test3"] = run_test3_verbalized_confidence(config)
    benchmark["test4"] = run_test4_textworld(config)
    episodes = run_test5_e2e(config, output_dir)
    benchmark["test5_episodes"] = len(episodes)
    benchmark["test6"] = run_test6_logging_analysis(episodes, output_dir)

    benchmark_path = output_dir / "pilot_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Wrote {benchmark_path}")

    calibration_path = output_dir / "pilot_calibration.json"
    with open(calibration_path, "w") as f:
        json.dump(episodes, f, indent=2)
    print(f"Wrote {calibration_path}")


if __name__ == "__main__":
    main()
