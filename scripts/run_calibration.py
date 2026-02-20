#!/usr/bin/env python3
"""
Difficulty calibration: generate TextWorld games at varying difficulty,
run Always-C0 on each, compute success rate per instance, output difficulty_manifest.json.
Usage: python scripts/run_calibration.py [--world-size 5 8] [--quest-length 2 4] [--num-instances 20] [--tracking-uri URI]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(config_path: str | Path) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_infra_env() -> None:
    """Load infra/.env so scripts use same MinIO credentials as dashboard."""
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / "infra" / ".env")
    except ImportError:
        pass


def _create_tracker(tracking_uri: str | None, infra_config: dict | None):
    if not tracking_uri and infra_config:
        tracking_uri = (infra_config.get("tracking") or {}).get("mlflow_uri") or ""
    if not tracking_uri or "<HOME_SERVER_IP>" in tracking_uri:
        return None
    try:
        from src.utils.experiment_tracker import ExperimentTracker
        s3 = (infra_config or {}).get("tracking", {}).get("s3_endpoint")
        exp_name = (infra_config or {}).get("tracking", {}).get("experiment_name", "metacog-llm-compute")
        access = os.environ.get("AWS_ACCESS_KEY_ID", "") or os.environ.get("MINIO_ROOT_USER", "")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "") or os.environ.get("MINIO_ROOT_PASSWORD", "")
        return ExperimentTracker(
            tracking_uri=tracking_uri,
            experiment_name=exp_name,
            s3_endpoint_url=s3,
            aws_access_key=access,
            aws_secret_key=secret,
        )
    except Exception as e:
        print(f"Warning: could not create experiment tracker: {e}")
        return None


def generate_game(
    output_dir: Path,
    instance_id: int,
    world_size: int,
    quest_length: int,
    seed: int,
) -> Path | None:
    """Generate one TextWorld game; return path to textworld_{instance_id}.ulx or None."""
    path = output_dir / f"textworld_{instance_id}.ulx"
    cmd = [
        sys.executable, "-m", "textworld.challenges",
        "custom",
        "--world-size", str(world_size),
        "--quest-length", str(quest_length),
        "--output", str(path),
        "--seed", str(seed),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(REPO_ROOT))
        return path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_c0_episode(env_factory, model, max_steps: int) -> bool:
    """Run one C0 episode; return task_success."""
    from src.agent.base_agent import run_episode
    from src.agent.compute_stages import get_step_fn
    env = env_factory()
    step_fn = get_step_fn("C0")
    result = run_episode(env, model, "C0", step_fn=step_fn, max_steps=max_steps)
    return result.get("task_success", False)


def main() -> None:
    parser = argparse.ArgumentParser(description="TextWorld difficulty calibration: C0 success rate per instance")
    parser.add_argument("--config", default="configs/experiment_core.yaml", help="Experiment config (for model/paths)")
    parser.add_argument("--output-dir", default="data/tasks/calibration", help="Directory for generated games and manifest")
    parser.add_argument("--world-size", type=int, nargs=2, default=[5, 8], metavar=("MIN", "MAX"), help="World size (rooms) range")
    parser.add_argument("--quest-length", type=int, nargs=2, default=[2, 4], metavar=("MIN", "MAX"), help="Quest length range")
    parser.add_argument("--num-instances", type=int, default=20, help="Number of game instances to generate and evaluate")
    parser.add_argument("--runs-per-instance", type=int, default=3, help="C0 runs per instance for success rate")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--infra-config", default="configs/infra.yaml", help="Infra YAML")
    parser.add_argument("--real", action="store_true", help="Use real model (otherwise mock)")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(REPO_ROOT / args.config) if (REPO_ROOT / args.config).exists() else {}
    max_steps = config.get("episode", {}).get("max_steps_per_episode", 20)

    infra_config = None
    if args.infra_config and (REPO_ROOT / args.infra_config).exists():
        _load_infra_env()
        infra_config = load_config(REPO_ROOT / args.infra_config)
    tracker = _create_tracker(args.tracking_uri, infra_config)

    # Model (mock or real)
    if args.real and config:
        from scripts.run_phase1 import _create_model
        model = _create_model(config, use_real=True)
    else:
        from scripts.run_phase1 import _MockModel
        model = _MockModel()

    ws_lo, ws_hi = args.world_size[0], args.world_size[1]
    ql_lo, ql_hi = args.quest_length[0], args.quest_length[1]

    if tracker:
        tracker.start_run(
            run_name=f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "world_size_range": [ws_lo, ws_hi],
                "quest_length_range": [ql_lo, ql_hi],
                "num_instances": args.num_instances,
                "runs_per_instance": args.runs_per_instance,
            },
            tags={"phase": "calibration"},
        )

    manifest = []
    for i in range(args.num_instances):
        world_size = ws_lo + (i % (ws_hi - ws_lo + 1))
        quest_length = ql_lo + (i % (ql_hi - ql_lo + 1))
        path = generate_game(output_dir, i, world_size, quest_length, seed=42 + i)
        if not path or not path.exists():
            print(f"  Skip instance {i}: game generation failed")
            continue
        # Run C0
        def make_env():
            from src.environments.textworld_env import TextWorldEnv
            return TextWorldEnv(game_file=str(path), max_steps=max_steps)
        successes = 0
        for _ in range(args.runs_per_instance):
            if run_c0_episode(make_env, model, max_steps):
                successes += 1
        rate = successes / args.runs_per_instance
        if rate > 0.85:
            tier = "easy"
        elif rate >= 0.40:
            tier = "medium"
        else:
            tier = "hard"
        entry = {
            "instance_id": i,
            "world_size": world_size,
            "quest_length": quest_length,
            "c0_success_rate": round(rate, 4),
            "tier": tier,
        }
        manifest.append(entry)
        if tracker:
            tracker.log_metrics({f"c0_success_instance_{i}": rate}, step=i)
        print(f"  Instance {i}: world_size={world_size}, quest_length={quest_length}, rate={rate:.2f}, tier={tier}")

    manifest_path = output_dir / "difficulty_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")

    if tracker:
        tracker.log_artifact(manifest_path, artifact_path="calibration")
        tracker.log_metrics({
            "num_instances": len(manifest),
            "easy_count": sum(1 for e in manifest if e["tier"] == "easy"),
            "medium_count": sum(1 for e in manifest if e["tier"] == "medium"),
            "hard_count": sum(1 for e in manifest if e["tier"] == "hard"),
        })
        tracker.end_run()


if __name__ == "__main__":
    main()
