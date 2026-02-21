#!/usr/bin/env python3
"""
Verify MLflow tracking connectivity and list experiments/runs.
Use this to confirm that runs from run_pilot.py are visible (same URI and experiment).
Usage: python scripts/check_mlflow_tracking.py [--infra-config configs/infra.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MLflow tracking connectivity and list runs.")
    parser.add_argument(
        "--infra-config",
        default="configs/infra.yaml",
        help="Infra YAML for tracking URI and experiment name",
    )
    args = parser.parse_args()

    # Load infra env (MinIO credentials)
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / "infra" / ".env")
    except ImportError:
        pass

    def load_config(path: str) -> dict:
        import yaml
        p = REPO_ROOT / path if not Path(path).is_absolute() else Path(path)
        if not p.exists():
            return {}
        with open(p) as f:
            return yaml.safe_load(f) or {}

    infra = load_config(args.infra_config)
    tracking = infra.get("tracking") or {}
    uri = tracking.get("mlflow_uri", "")
    exp_name = tracking.get("experiment_name", "metacog-llm-compute")

    if not uri or "<HOME_SERVER_IP>" in uri:
        print("No valid mlflow_uri in infra config. Set tracking.mlflow_uri in configs/infra.yaml (e.g. http://mlflow.home)")
        sys.exit(1)

    import os
    os.environ["MLFLOW_TRACKING_URI"] = uri
    s3 = tracking.get("s3_endpoint")
    if s3:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3

    print(f"Tracking URI: {uri}")
    print(f"Experiment name: {exp_name}")
    print()

    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("mlflow is not installed. pip install mlflow")
        sys.exit(1)

    client = MlflowClient()
    try:
        experiments = client.search_experiments()
    except Exception as e:
        print(f"Failed to list experiments: {e}")
        print("Check that mlflow.home is reachable from this machine (DNS/network) and the MLflow server is running.")
        sys.exit(1)

    print(f"Experiments ({len(experiments)}):")
    for e in experiments:
        print(f"  - {e.name!r} (id={e.experiment_id})")
    print()

    exp = client.get_experiment_by_name(exp_name)
    if not exp:
        print(f"Experiment {exp_name!r} not found. Create a run first (e.g. run_pilot.py with --infra-config).")
        sys.exit(0)

    try:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=20,
        )
    except Exception as e:
        print(f"Failed to list runs: {e}")
        sys.exit(1)

    print(f"Runs in {exp_name!r} (most recent 20):")
    if not runs:
        print("  (no runs)")
        print()
        print("If you just ran the pilot, ensure you used the same infra config and that no 'Warning: could not create experiment tracker' appeared.")
        sys.exit(0)

    for r in runs:
        print(f"  - {r.info.run_name or r.info.run_id}  id={r.info.run_id}  status={r.info.status}")
    print()
    print("If you see your run here but not in the MLflow UI:")
    print("  1. In the MLflow UI (e.g. http://mlflow.home), use the experiment dropdown and select", repr(exp_name))
    print("  2. Ensure you are on the same network and that mlflow.home resolves to the same server")


if __name__ == "__main__":
    main()
