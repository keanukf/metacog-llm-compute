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

    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify backend logprob parity (§5.7)")
    parser.add_argument("--config", default="configs/experiment_core.yaml")
    parser.add_argument("--probes", default="data/probes/parity_prompts.json")
    parser.add_argument("--backend", choices=["vllm", "lmstudio", "mock", "server"], default="mock")
    parser.add_argument("--compare-backends", action="store_true")
    parser.add_argument("--output-dir", default="data/results")
    args = parser.parse_args()

    import yaml

    config_path = REPO_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    probes = _load_probes(REPO_ROOT / args.probes)

    from src.utils.experiment_env import create_experiment_model

    if args.backend == "server":
        from src.execution.backend.factory import create_execution_backend

        model = create_execution_backend(config, use_real=True)
    else:
        config.setdefault("inference", {})["backend"] = args.backend
        model = create_experiment_model(config, use_real=args.backend != "mock")
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "probes_file": str(args.probes),
        "backends": [_check_backend(model, probes, label=args.backend)],
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
    all_pass = b0["k_coverage_pass"] and b0["temperature_invariance_pass"]
    print(f"K-coverage: {'PASS' if b0['k_coverage_pass'] else 'FAIL'}")
    print(
        f"Temperature invariance: {'PASS' if b0['temperature_invariance_pass'] else 'FAIL'} "
        f"(eps={b0['temperature_invariance_eps_bits']} bits)"
    )
    raise SystemExit(0 if all_pass or args.backend == "mock" else 1)


if __name__ == "__main__":
    main()
