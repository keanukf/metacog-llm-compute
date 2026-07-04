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
                    # use first action token row heuristic
                    if not rows:
                        continue
                    top = rows[0].get("top_logprobs") if isinstance(rows[0], dict) else None
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
