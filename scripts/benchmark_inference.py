#!/usr/bin/env python3
"""
Standalone inference speed benchmark (same logic as pilot Test 1).

Measures output tokens/s and latency using the configured model wrapper.
Reads model and inference settings from the given YAML (e.g. configs/experiment_core.yaml).

Usage (from repo root):
  python scripts/benchmark_inference.py --config configs/experiment_core.yaml --pilot-mode hf
  python scripts/benchmark_inference.py --config configs/pilot.yaml --pilot-mode lmstudio
  python scripts/benchmark_inference.py --config configs/pilot.yaml --real --output-dir data/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import pilot helpers from the same directory as this script
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import run_pilot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LM inference speed (tokens/s, latency). Uses pilot Test 1 harness."
    )
    parser.add_argument("--config", default="configs/pilot.yaml", help="YAML with model, inference, optional test1_inference")
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Directory for inference_benchmark.json (created if missing)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Override JSON path (default: <output-dir>/inference_benchmark.json)",
    )
    parser.add_argument(
        "--pilot-mode",
        type=run_pilot.parse_pilot_mode_arg,
        default="mock",
        help="mock | hf | m1 (deprecated=hf) | cuda | litellm | lmstudio",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Auto-pick hf vs cuda when pilot-mode is mock (same as run_pilot.py)",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    output_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = run_pilot.load_config(config_path)
    print(f"Config: {config_path}")

    pilot_mode = run_pilot._resolve_pilot_mode(args)
    print(f"Pilot mode: {pilot_mode}")
    if pilot_mode != "mock":
        model_name = config.get("model", {}).get("name", "?")
        print(f"Loading model: {model_name} ...")
    real_model = run_pilot._create_real_model(config, pilot_mode)
    if pilot_mode != "mock" and real_model is None:
        print("Warning: real model could not be created; falling back to mock benchmark.")
        pilot_mode = "mock"
    elif pilot_mode != "mock":
        print("Model ready.")

    result = run_pilot.run_test1_inference_speed(config, output_dir, real_model=real_model)
    out: dict = {
        "pilot_mode": pilot_mode,
        "config_path": str(config_path),
        **result,
    }

    json_path = Path(args.output_json) if args.output_json else output_dir / "inference_benchmark.json"
    if not json_path.is_absolute():
        json_path = REPO_ROOT / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"tokens/s: {result.get('tokens_per_sec', 0):.2f}")
    print(f"latency mean (s): {result.get('latency_mean', 0):.4f}  std: {result.get('latency_std', 0):.4f}")
    print(f"total output tokens: {result.get('total_tokens', 0)}  prompts: {result.get('num_prompts', 0)}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
