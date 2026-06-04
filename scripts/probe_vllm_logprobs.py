#!/usr/bin/env python3
"""
Probe vLLM / CUDA backend logprob output (L0.1 sanity).

Runs one short ``generate(logprobs=True)`` call and prints token coverage stats.
Use on RunPod after ``setup_cloud.sh`` or locally with ``--pilot-mode cuda --real``.

Example:
  python scripts/probe_vllm_logprobs.py --config configs/pilot.yaml --pilot-mode cuda --real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.signals.token_entropy import compute_tle, extract_action_tle_from_response  # noqa: E402
from src.utils.inference.logprobs import logprob_token_coverage  # noqa: E402
from src.utils.pilot_config import load_yaml_path  # noqa: E402


def _short_prompt() -> str:
    return (
        "You are in a text adventure. Exits: north.\n"
        "Reply with exactly one game command on a single line (example: go north). "
        "No explanation."
    )


def _create_real_model(config: dict[str, Any], pilot_mode: str):
    from scripts.run_pilot import _create_real_model as create  # noqa: WPS433

    return create(config, pilot_mode=pilot_mode)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    p.add_argument("--pilot-mode", type=str, default="cuda")
    p.add_argument("--real", action="store_true", help="Require a real backend (no mock)")
    p.add_argument("--max-tokens", type=int, default=32)
    args = p.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    config = load_yaml_path(config_path)
    pilot_mode = args.pilot_mode.strip().lower()
    if args.real and pilot_mode == "mock":
        print("error: --real requires a non-mock pilot mode", file=sys.stderr)
        return 2

    try:
        model, err = _create_real_model(config, pilot_mode=pilot_mode)
    except Exception as exc:
        print(f"error: could not create model wrapper: {exc}", file=sys.stderr)
        return 1
    if model is None:
        detail = f" ({err})" if err else ""
        print(f"error: model wrapper is None{detail}", file=sys.stderr)
        return 1

    inf = config.get("inference", {}) or {}
    temperature = float(inf.get("temperature", 0.3))
    text, logprobs = model.generate(
        _short_prompt(),
        logprobs=True,
        max_tokens=int(args.max_tokens),
        temperature=temperature,
        enable_thinking=False,
    )

    coverage = logprob_token_coverage(logprobs)
    tle_full = compute_tle(logprobs) if logprobs else None
    tle_action = extract_action_tle_from_response(text or "", logprobs)

    report: dict[str, Any] = {
        "pilot_mode": pilot_mode,
        "model": (config.get("model") or {}).get("name"),
        "completion_text": text or "",
        "has_logprobs": bool(logprobs),
        "logprob_coverage": coverage,
        "tle_full_completion": tle_full,
        "tle_action_slice": tle_action,
        "first_three_logprob_records": (logprobs or [])[:3],
    }

    if not logprobs:
        print("warning: generate returned no logprobs", file=sys.stderr)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    ok = bool(logprobs) and coverage.get("n_tokens", 0) > 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
