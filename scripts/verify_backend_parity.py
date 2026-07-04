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
) -> dict[str, Any]:
    k_ok = True
    temp_ok = True
    details: list[dict[str, Any]] = []
    for probe in probes:
        prompt = probe["prompt"]
        text0, lp0 = model.generate(
            prompt, logprobs=True, temperature=0.3, enable_thinking=False, max_tokens=8
        )
        text1, lp1 = model.generate(
            prompt, logprobs=True, temperature=1.0, enable_thinking=False, max_tokens=8
        )
        counts0 = _topk_from_logprobs(lp0)
        if counts0 and min(counts0) < min_k:
            k_ok = False
        # Temperature invariance on first action-token row
        if lp0 and lp1 and isinstance(lp0[0], dict) and isinstance(lp1[0], dict):
            t0 = lp0[0].get("top_logprobs")
            t1 = lp1[0].get("top_logprobs")
            if json.dumps(t0, sort_keys=True) != json.dumps(t1, sort_keys=True):
                temp_ok = False
        details.append(
            {
                "id": probe.get("id"),
                "n_tokens": len(lp0 or []),
                "min_topk": min(counts0) if counts0 else 0,
                "entropies": _entropies(lp0),
            }
        )
    return {
        "backend": label,
        "k_coverage_pass": k_ok,
        "temperature_invariance_pass": temp_ok,
        "entropy_equality_pass": "not_applicable",
        "entropy_equality_note": "Cross-backend entropy equality requires identical model+precision",
        "probes": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify backend logprob parity (§5.7)")
    parser.add_argument("--config", default="configs/experiment_core.yaml")
    parser.add_argument("--probes", default="data/probes/parity_prompts.json")
    parser.add_argument("--backend", choices=["vllm", "lmstudio", "mock"], default="mock")
    parser.add_argument("--compare-backends", action="store_true")
    parser.add_argument("--output-dir", default="data/results")
    args = parser.parse_args()

    import yaml

    config_path = REPO_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    probes = _load_probes(REPO_ROOT / args.probes)

    from src.utils.experiment_env import create_experiment_model

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
    print(f"Temperature invariance: {'PASS' if b0['temperature_invariance_pass'] else 'FAIL'}")
    raise SystemExit(0 if all_pass or args.backend == "mock" else 1)


if __name__ == "__main__":
    main()
