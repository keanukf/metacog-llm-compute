#!/usr/bin/env python3
"""K ∈ {5,10,20} TLE sensitivity from stored logprob sidecars (Phase 0)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.calibration import _label_from_correctness, compute_auroc
from src.analysis.datasets import load_run_dataset
from src.signals.token_entropy import tle_mean_entropy_at_k_from_logprob_tokens


def _step_logprob_token_lists(step_entry: dict[str, Any] | None) -> list[list[dict[str, Any]]]:
    """Return one or more candidate logprob token lists for an env step sidecar entry."""
    if not isinstance(step_entry, dict):
        return []
    if isinstance(step_entry.get("samples"), list):
        out: list[list[dict[str, Any]]] = []
        for sample in step_entry["samples"]:
            if not isinstance(sample, dict):
                continue
            toks = sample.get("logprob_tokens")
            if isinstance(toks, list) and toks:
                out.append([t for t in toks if isinstance(t, dict)])
        return out
    toks = step_entry.get("logprob_tokens")
    if isinstance(toks, list) and toks:
        return [[t for t in toks if isinstance(t, dict)]]
    return []


def _sidecar_step_entry(payload: object, step_index: int | None) -> dict[str, Any] | None:
    if isinstance(payload, list):
        return {"step_index": step_index, "logprob_tokens": payload}
    if not isinstance(payload, dict):
        return None
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None
    if step_index is not None:
        for step in steps:
            if isinstance(step, dict) and step.get("step_index") == step_index:
                return step
    if steps and isinstance(steps[0], dict):
        return steps[0]
    return None


def _pick_logprob_tokens(
    candidates: list[list[dict[str, Any]]],
    *,
    reference_mean_entropy: float | None,
    k: int,
) -> list[dict[str, Any]] | None:
    """Pick C2 sample whose action-window mean entropy at K best matches episode TLE."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if reference_mean_entropy is None:
        return candidates[0]
    best: list[dict[str, Any]] | None = None
    best_diff = float("inf")
    for toks in candidates:
        ent = tle_mean_entropy_at_k_from_logprob_tokens(toks, k)
        if ent is None or math.isnan(ent):
            continue
        diff = abs(float(ent) - float(reference_mean_entropy))
        if diff < best_diff:
            best_diff = diff
            best = toks
    return best if best is not None else candidates[0]


def _tle_score_at_k(
    payload: object,
    step_index: int | None,
    k: int,
    *,
    reference_mean_entropy: float | None = None,
) -> float | None:
    """Committed-action mean entropy at K; AUROC uses ``-`` this value as score."""
    step_entry = _sidecar_step_entry(payload, step_index)
    candidates = _step_logprob_token_lists(step_entry)
    toks = _pick_logprob_tokens(
        candidates,
        reference_mean_entropy=reference_mean_entropy,
        k=k,
    )
    if not toks:
        return None
    return tle_mean_entropy_at_k_from_logprob_tokens(toks, k)


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
                payload = json.loads(Path(lp_path).read_text(encoding="utf-8"))
                for sd in ep.get("steps_detail") or []:
                    if not isinstance(sd, dict):
                        continue
                    y01 = _label_from_correctness(sd.get("correctness"), "optimal_only")
                    if y01 is None:
                        continue
                    step_index_raw = sd.get("step_index")
                    step_index = int(step_index_raw) if step_index_raw is not None else None
                    ref = None
                    tle = sd.get("tle")
                    if isinstance(tle, dict) and isinstance(tle.get("mean_entropy"), (int, float)):
                        ref = float(tle["mean_entropy"])
                    ent = _tle_score_at_k(
                        payload,
                        step_index,
                        k,
                        reference_mean_entropy=ref,
                    )
                    if ent is None or math.isnan(ent):
                        continue
                    scores.append(-float(ent))
                    labels.append(1 if y01 >= 0.5 else 0)
            auroc = compute_auroc(scores, labels) if scores else float("nan")
            report[dom][str(k)] = {"auroc": auroc, "n": len(scores)}

    out = Path(args.output) if args.output else run_dir / "topk_sensitivity.json"
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
