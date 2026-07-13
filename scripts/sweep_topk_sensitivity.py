#!/usr/bin/env python3
"""K ∈ {5,10,20} TLE sensitivity from stored logprob sidecars (Phase 0)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.calibration import compute_auroc
from src.analysis.datasets import load_run_dataset
from src.signals.token_entropy import entropy_shannon_from_top_logprobs


def _tle_for_k(top_logprobs: list[dict], k: int) -> float | None:
    if not top_logprobs:
        return None
    trimmed = top_logprobs[:k]
    try:
        return float(entropy_shannon_from_top_logprobs(trimmed))
    except Exception:
        return None


def _action_top_logprobs(payload: object, step_index: int | None) -> list[dict] | None:
    """First committed-action token top_logprobs for an env step (sidecar v1/v2)."""
    if isinstance(payload, list):
        token_rows = [r for r in payload if isinstance(r, dict)]
    elif isinstance(payload, dict):
        steps = payload.get("steps")
        if not isinstance(steps, list):
            return None
        step_entry = None
        if step_index is not None:
            for step in steps:
                if isinstance(step, dict) and step.get("step_index") == step_index:
                    step_entry = step
                    break
        if step_entry is None and steps and isinstance(steps[0], dict):
            step_entry = steps[0]
        if not isinstance(step_entry, dict):
            return None
        token_rows = step_entry.get("logprob_tokens")
        if not isinstance(token_rows, list) and isinstance(step_entry.get("samples"), list):
            samples = step_entry["samples"]
            if samples and isinstance(samples[0], dict):
                token_rows = samples[0].get("logprob_tokens")
        if not isinstance(token_rows, list):
            return None
        token_rows = [r for r in token_rows if isinstance(r, dict)]
    else:
        return None
    if not token_rows:
        return None
    top = token_rows[0].get("top_logprobs")
    return top if isinstance(top, list) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-K TLE sensitivity sweep")
    parser.add_argument("run_dir", help="Pilot/phase1 run dir with logprob sidecars")
    parser.add_argument("--output", default=None)
    parser.add_argument("--ks", default="5,10,20")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    ks = [int(x) for x in str(args.ks).split(",") if x.strip()]
    ds = load_run_dataset(run_dir)
    report: dict[str, dict] = {}
    for dom in sorted({str(e.get("domain")) for e in ds.episodes}):
        report[dom] = {}
        for k in ks:
            scores: list[float] = []
            labels: list[int] = []
            for ep in ds.episodes:
                if str(ep.get("domain")) != dom:
                    continue
                lp_path = ep.get("logprobs_json_path")
                if not lp_path or not Path(lp_path).is_file():
                    continue
                rows = json.loads(Path(lp_path).read_text(encoding="utf-8"))
                for sd in ep.get("steps_detail") or []:
                    y = sd.get("correctness")
                    if y != "optimal":
                        label = 0
                    elif y == "optimal":
                        label = 1
                    else:
                        continue
                    step_index_raw = sd.get("step_index")
                    step_index = int(step_index_raw) if step_index_raw is not None else None
                    top = _action_top_logprobs(rows, step_index)
                    if not isinstance(top, list):
                        continue
                    ent = _tle_for_k(top, k)
                    if ent is None or math.isnan(ent):
                        continue
                    scores.append(-ent)
                    labels.append(label)
            auroc = compute_auroc(scores, labels) if scores else float("nan")
            report[dom][str(k)] = {"auroc": auroc, "n": len(scores)}

    out = Path(args.output) if args.output else run_dir / "topk_sensitivity.json"
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
