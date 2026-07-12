#!/usr/bin/env python3
"""Backend logprob parity checks (thesis §5.7)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.execution.parity import run_batch_invariance_probe
from src.signals.token_entropy import entropy_shannon_from_top_logprobs
from src.utils.inference.logprob_invariance import (
    TLE_INVARIANCE_EPS_BITS,
    TLE_INVARIANCE_NOISE_SAFETY_FACTOR,
    probe_temperature_invariance,
    resolve_tle_invariance_eps,
)


def _load_probes(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("parity_prompts.json must be a list")
    return [x for x in raw if isinstance(x, dict) and x.get("prompt")]


def _topk_from_logprobs(logprobs: list[dict[str, Any]] | None) -> list[int]:
    if not logprobs:
        return []
    counts: list[int] = []
    for row in logprobs:
        if not isinstance(row, dict):
            counts.append(0)
            continue
        top = row.get("top_logprobs")
        counts.append(len(top) if isinstance(top, list) else 0)
    return counts


def _entropies(logprobs: list[dict[str, Any]] | None) -> list[float]:
    if not logprobs:
        return []
    out: list[float] = []
    for row in logprobs:
        if not isinstance(row, dict):
            continue
        top = row.get("top_logprobs")
        if not isinstance(top, list):
            continue
        try:
            out.append(float(entropy_shannon_from_top_logprobs(top)))
        except Exception:
            continue
    return out


def _check_backend(
    model: Any,
    probes: list[dict[str, str]],
    *,
    label: str,
    min_k: int = 20,
    t_low: float = 0.3,
    t_high: float = 1.0,
    max_concurrent_episodes: int | None = None,
    run_batch_invariance: bool = False,
) -> dict[str, Any]:
    k_ok = True
    temp_ok = True
    details: list[dict[str, Any]] = []
    same_t_dtle_values: list[float] = []

    for probe in probes:
        prompt = probe["prompt"]
        diag = probe_temperature_invariance(
            model,
            prompt,
            t_low=t_low,
            t_high=t_high,
            max_tokens=8,
        )
        same = diag.get("same_t_dtle")
        if same is not None:
            same_t_dtle_values.append(float(same))

        counts0 = _topk_from_logprobs(diag.get("logprobs_t_low"))
        if counts0 and min(counts0) < min_k:
            k_ok = False

        details.append(
            {
                "id": probe.get("id"),
                "n_tokens": len(diag.get("logprobs_t_low") or []),
                "min_topk": min(counts0) if counts0 else 0,
                "entropies": _entropies(diag.get("logprobs_t_low")),
                "temperature_invariance": diag,
            }
        )

    eps = resolve_tle_invariance_eps(same_t_dtle_values)
    for row in details:
        inv = row.get("temperature_invariance") or {}
        cross = inv.get("cross_t_dtle")
        if cross is None or float(cross) > eps:
            temp_ok = False
        inv["pass"] = cross is not None and float(cross) <= eps

    result: dict[str, Any] = {
        "backend": label,
        "k_coverage_pass": k_ok,
        "temperature_invariance_pass": temp_ok,
        "temperature_invariance_eps_bits": eps,
        "temperature_invariance_preregistered_floor_bits": TLE_INVARIANCE_EPS_BITS,
        "temperature_invariance_noise_safety_factor": TLE_INVARIANCE_NOISE_SAFETY_FACTOR,
        "temperature_invariance_same_t_dtle_max": (
            max(same_t_dtle_values) if same_t_dtle_values else None
        ),
        "temperature_invariance_note": (
            "Pass when |dTLE(T_low vs T_high)| <= eps on first-token top-k; "
            "eps = max(preregistered floor, same-T noise floor * safety factor). "
            "Same-T control and predicted scaling spans are diagnostic only."
        ),
        "entropy_equality_pass": "not_applicable",
        "entropy_equality_note": "Cross-backend entropy equality requires identical model+precision",
        "probes": details,
    }

    if run_batch_invariance and max_concurrent_episodes is not None:
        batch = run_batch_invariance_probe(
            model,
            probes,
            max_concurrent_episodes=max_concurrent_episodes,
            eps=eps,
        )
        result["batch_invariance_pass"] = bool(batch["passed"])
        result["batch_invariance"] = batch
        result["batch_invariance_note"] = (
            "Pass when |dTLE(solo vs under concurrent server load)| <= eps at the "
            "committed-action TLE window; load uses saturated pool at production N."
        )
    else:
        result["batch_invariance_pass"] = "not_applicable"
        result["batch_invariance_note"] = (
            "Batch invariance runs only with --backend server (parallel vLLM)."
        )

    return result


def _all_pass(backend_report: dict[str, Any], *, backend_label: str) -> bool:
    if backend_label == "mock":
        return True
    batch = backend_report.get("batch_invariance_pass")
    batch_ok = batch is True or batch == "not_applicable"
    return (
        bool(backend_report.get("k_coverage_pass"))
        and bool(backend_report.get("temperature_invariance_pass"))
        and batch_ok
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify backend logprob parity (§5.7)")
    parser.add_argument("--config", default="configs/experiment_core.yaml")
    parser.add_argument("--probes", default="data/probes/parity_prompts.json")
    parser.add_argument("--backend", choices=["vllm", "lmstudio", "mock", "server"], default="mock")
    parser.add_argument("--compare-backends", action="store_true")
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument(
        "--freeze-metadata-dir",
        default=None,
        help="If set, write frozen (N, eps) into run_metadata.json under this checkpoint dir",
    )
    args = parser.parse_args()

    import yaml

    from src.execution.config import (
        ExecutionConfig,
        frozen_execution_params_dict,
        write_frozen_execution_params,
    )

    config_path = REPO_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    probes = _load_probes(REPO_ROOT / args.probes)
    exec_cfg = ExecutionConfig.from_config(config, real=args.backend == "server")

    from src.utils.experiment_env import create_experiment_model

    close_fn: Any = None
    run_batch = args.backend == "server"
    max_n = exec_cfg.max_concurrent_episodes if run_batch else None

    if args.backend == "server":
        from src.execution.backend.factory import create_execution_backend

        model = create_execution_backend(config, use_real=True)
        close_fn = getattr(model, "close", None)
    else:
        config.setdefault("inference", {})["backend"] = args.backend
        model = create_experiment_model(config, use_real=args.backend != "mock")

    try:
        backend_report = _check_backend(
            model,
            probes,
            label=args.backend,
            max_concurrent_episodes=max_n,
            run_batch_invariance=run_batch,
        )
    finally:
        if close_fn is not None:
            close_fn()

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "probes_file": str(args.probes),
        "config": str(args.config),
        "max_concurrent_episodes": max_n,
        "backends": [backend_report],
    }
    if args.compare_backends:
        report["note"] = "Run separately on each backend host; merge JSON for thesis §5.7.5"

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"backend_parity_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    b0 = report["backends"][0]
    all_pass = _all_pass(b0, backend_label=args.backend)
    print(f"K-coverage: {'PASS' if b0['k_coverage_pass'] else 'FAIL'}")
    print(
        f"Temperature invariance: {'PASS' if b0['temperature_invariance_pass'] else 'FAIL'} "
        f"(eps={b0['temperature_invariance_eps_bits']} bits)"
    )
    batch_pass = b0.get("batch_invariance_pass")
    if batch_pass == "not_applicable":
        print("Batch invariance: not_applicable (use --backend server on vLLM pod)")
    else:
        batch = b0.get("batch_invariance") or {}
        print(
            f"Batch invariance: {'PASS' if batch_pass else 'FAIL'} "
            f"(max_dtle={batch.get('max_dtle')}, eps={batch.get('eps')} bits)"
        )

    if args.freeze_metadata_dir is not None and run_batch and batch_pass is True:
        meta_dir = Path(args.freeze_metadata_dir)
        if not meta_dir.is_absolute():
            meta_dir = REPO_ROOT / meta_dir
        batch_eps = float(
            (b0.get("batch_invariance") or {}).get("eps", b0["temperature_invariance_eps_bits"])
        )
        write_frozen_execution_params(
            meta_dir,
            frozen_execution_params_dict(
                max_concurrent_episodes=exec_cfg.max_concurrent_episodes,
                tle_invariance_eps=batch_eps,
                eps_derived_under_load=True,
            ),
        )
        print(f"Froze execution params in {meta_dir / 'run_metadata.json'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
