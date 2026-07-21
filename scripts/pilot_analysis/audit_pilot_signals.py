#!/usr/bin/env python3
"""
Audit pilot run folders for L0.1 / L2 signal coverage and C2 trace shape.

Reports VC/TLE rates overall and per compute stage, sanity logprob fields,
feasibility summary, and C2 majority-vote trace checks.

Example:
  python scripts/pilot_analysis/audit_pilot_signals.py data/results/runpod_pilot/pilot_20250604_120000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pilot_analysis.validate_pilot_outputs import (  # noqa: E402
    _iter_jsonl,
    _read_json,
    validate,
)


def _signal_rates_by_stage(
    episode_jsons: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate VC/TLE non-null rates grouped by stage_per_step entry."""
    buckets: dict[str, dict[str, int]] = {}
    for ep in episode_jsons:
        stages = ep.get("stage_per_step") or []
        vcs = ep.get("vc_per_step") or []
        tles = ep.get("tle_per_step") or []
        n = max(len(stages), len(vcs), len(tles))
        for i in range(n):
            stage = str(stages[i] if i < len(stages) else "unknown")
            b = buckets.setdefault(
                stage, {"vc_total": 0, "vc_nonnull": 0, "tle_total": 0, "tle_nonnull": 0}
            )
            if i < len(vcs):
                b["vc_total"] += 1
                if vcs[i] is not None:
                    b["vc_nonnull"] += 1
            if i < len(tles):
                b["tle_total"] += 1
                if tles[i] is not None:
                    b["tle_nonnull"] += 1
    out: dict[str, dict[str, float | int]] = {}
    for stage, b in sorted(buckets.items()):
        vc_total = int(b["vc_total"])
        tle_total = int(b["tle_total"])
        out[stage] = {
            "vc_total": vc_total,
            "vc_nonnull": int(b["vc_nonnull"]),
            "vc_rate": (float(b["vc_nonnull"]) / vc_total) if vc_total else None,
            "tle_total": tle_total,
            "tle_nonnull": int(b["tle_nonnull"]),
            "tle_rate": (float(b["tle_nonnull"]) / tle_total) if tle_total else None,
        }
    return out


def _audit_c2_traces(pilot_dir: Path) -> dict[str, Any]:
    """Scan trace JSONL for C2 steps and verify majority-vote call_detail shape."""
    c2_steps = 0
    ok_method = 0
    sample_counts: list[int] = []
    methods: Counter[str] = Counter()
    for tf in sorted(pilot_dir.glob("trace_ep_*.jsonl")):
        for row in _iter_jsonl(tf):
            stage = str(row.get("compute_stage") or row.get("stage") or "")
            if stage != "C2":
                continue
            c2_steps += 1
            detail = row.get("call_detail") or {}
            if not isinstance(detail, dict):
                continue
            method = str(detail.get("method") or "")
            methods[method] += 1
            if method == "self_consistency_majority_vote":
                ok_method += 1
            subcalls = detail.get("subcalls") or []
            if isinstance(subcalls, list):
                n_samples = sum(
                    1 for s in subcalls if isinstance(s, dict) and s.get("kind") == "sample"
                )
                if n_samples:
                    sample_counts.append(n_samples)
    return {
        "c2_steps_seen": c2_steps,
        "c2_steps_with_majority_vote_method": ok_method,
        "c2_method_counts": dict(methods),
        "c2_sample_subcall_counts": sample_counts,
    }


def _episode_success_summary(episode_jsons: list[dict[str, Any]]) -> dict[str, Any]:
    tw = [e for e in episode_jsons if str(e.get("domain") or "").startswith("textworld")]
    toh = [e for e in episode_jsons if "hanoi" in str(e.get("domain") or "").lower()]
    return {
        "textworld_episodes": len(tw),
        "textworld_success": sum(1 for e in tw if e.get("success") is True),
        "tower_of_hanoi_episodes": len(toh),
        "tower_of_hanoi_success": sum(1 for e in toh if e.get("success") is True),
    }


def audit(pilot_dir: Path) -> dict[str, Any]:
    ep_jsons: list[dict[str, Any]] = []
    for pattern in ("ep_textworld_*.json", "ep_tower_of_hanoi_*.json"):
        ep_jsons.extend([_read_json(p) or {} for p in sorted(pilot_dir.glob(pattern))])
    ep_jsons = [e for e in ep_jsons if e]

    report: dict[str, Any] = {
        "pilot_dir": str(pilot_dir),
        "sanity": _read_json(pilot_dir / "pilot_sanity.json"),
        "feasibility": _read_json(pilot_dir / "pilot_feasibility.json"),
        "test5_toh": _read_json(pilot_dir / "pilot_test5_toh.json"),
        "signals_by_stage": _signal_rates_by_stage(ep_jsons),
        "episode_success": _episode_success_summary(ep_jsons),
        "c2_trace_audit": _audit_c2_traces(pilot_dir),
    }

    ok, errors = validate(pilot_dir)
    report["validate_pilot_outputs_ok"] = ok
    report["validate_pilot_outputs_errors"] = errors
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pilot_dir", type=Path, help="Path to pilot_YYYYMMDD_HHMMSS folder")
    p.add_argument("--json", action="store_true", help="Print full JSON report")
    args = p.parse_args()

    pilot_dir = args.pilot_dir
    if not pilot_dir.is_absolute():
        pilot_dir = (REPO_ROOT / pilot_dir).resolve()
    if not pilot_dir.is_dir():
        print(f"error: not a directory: {pilot_dir}", file=sys.stderr)
        return 2

    report = audit(pilot_dir)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Pilot dir: {report['pilot_dir']}")
        san = report.get("sanity") or {}
        if san:
            print(
                "Sanity:",
                f"has_logprobs={san.get('has_logprobs')}",
                f"tokens={san.get('completion_tokens_observed')}",
                f"token_field_rate={san.get('logprob_token_field_rate')}",
            )
        print("Signals by stage:")
        for stage, stats in (report.get("signals_by_stage") or {}).items():
            print(
                f"  {stage}: vc_rate={stats.get('vc_rate')} "
                f"tle_rate={stats.get('tle_rate')} (n={stats.get('tle_total')})"
            )
        c2 = report.get("c2_trace_audit") or {}
        if c2.get("c2_steps_seen"):
            print(
                "C2 traces:",
                f"steps={c2.get('c2_steps_seen')}",
                f"majority_vote_method={c2.get('c2_steps_with_majority_vote_method')}",
                f"sample_counts={c2.get('c2_sample_subcall_counts')}",
            )
        ep = report.get("episode_success") or {}
        print(
            "Episodes:",
            f"TW success {ep.get('textworld_success')}/{ep.get('textworld_episodes')}",
            f"ToH success {ep.get('tower_of_hanoi_success')}/{ep.get('tower_of_hanoi_episodes')}",
        )
        ok = report.get("validate_pilot_outputs_ok")
        print(f"validate_pilot_outputs: {'OK' if ok else 'FAIL'}")
        for err in report.get("validate_pilot_outputs_errors") or []:
            print(f"  - {err}")

    return 0 if report.get("validate_pilot_outputs_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
