#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _episode_signal_rates(episode_jsons: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    total_vc = 0
    nonnull_vc = 0
    total_tle = 0
    nonnull_tle = 0
    for ep in episode_jsons:
        for v in ep.get("vc_per_step") or []:
            total_vc += 1
            if v is not None:
                nonnull_vc += 1
        for t in ep.get("tle_per_step") or []:
            total_tle += 1
            if t is not None:
                nonnull_tle += 1
    vc_rate = (nonnull_vc / total_vc) if total_vc else None
    tle_rate = (nonnull_tle / total_tle) if total_tle else None
    return vc_rate, tle_rate


def validate(pilot_dir: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []

    san_path = pilot_dir / "pilot_sanity.json"
    san = _read_json(san_path)
    if san is not None:
        if san.get("has_logprobs") is not True:
            errors.append("sanity: has_logprobs is not true")
        cto = san.get("completion_tokens_observed")
        if not isinstance(cto, int) or cto <= 0:
            errors.append(f"sanity: completion_tokens_observed expected int>0, got {cto!r}")

    t2_path = pilot_dir / "pilot_test2_tle.json"
    t2 = _read_json(t2_path)
    if t2 is not None:
        mean_ent = ((t2.get("summary") or {}) if isinstance(t2.get("summary"), dict) else {}).get(
            "mean_entropy_avg"
        )
        if not isinstance(mean_ent, (int, float)):
            errors.append(f"test2: summary.mean_entropy_avg expected float, got {mean_ent!r}")

    # Episodes
    ep_jsons: list[dict[str, Any]] = []
    ep_jsons.extend([_read_json(p) or {} for p in sorted(pilot_dir.glob("ep_textworld_*.json"))])
    ep_jsons.extend([_read_json(p) or {} for p in sorted(pilot_dir.glob("ep_tower_of_hanoi_*.json"))])
    ep_jsons = [e for e in ep_jsons if isinstance(e, dict) and e]

    if ep_jsons:
        vc_rate, tle_rate = _episode_signal_rates(ep_jsons)
        if vc_rate is None:
            errors.append("episodes: vc_rate missing (no vc_per_step data found)")
        elif vc_rate < 0.80:
            errors.append(f"episodes: vc_rate {vc_rate:.3f} < 0.80")
        if tle_rate is None:
            errors.append("episodes: tle_rate missing (no tle_per_step data found)")
        elif tle_rate < 0.95:
            errors.append(f"episodes: tle_rate {tle_rate:.3f} < 0.95")

    t5_path = pilot_dir / "pilot_test5_toh.json"
    t5 = _read_json(t5_path)
    if t5 is not None:
        parse_rate = t5.get("parse_rate")
        legal_rate = t5.get("avg_legal_rate")
        if not isinstance(parse_rate, (int, float)):
            errors.append(f"test5: parse_rate expected float, got {parse_rate!r}")
        elif float(parse_rate) < 0.95:
            errors.append(f"test5: parse_rate {float(parse_rate):.3f} < 0.95")
        if not isinstance(legal_rate, (int, float)):
            errors.append(f"test5: avg_legal_rate expected float, got {legal_rate!r}")
        elif float(legal_rate) < 0.95:
            errors.append(f"test5: avg_legal_rate {float(legal_rate):.3f} < 0.95")

    # Traces: ensure response_full exists and is non-empty
    trace_files = sorted(pilot_dir.glob("trace_ep_*.jsonl"))
    if trace_files:
        for tf in trace_files:
            rows = _iter_jsonl(tf)
            if not rows:
                errors.append(f"traces: {tf.name} had no readable jsonl rows")
                continue
            for i, row in enumerate(rows):
                resp = row.get("response_full")
                if not isinstance(resp, str) or len(resp.strip()) == 0:
                    errors.append(f"traces: {tf.name} row {i} missing/empty response_full")
                    break

    return (len(errors) == 0), errors


def main() -> None:
    p = argparse.ArgumentParser(description="Validate a pilot_YYYYMMDD_HHMMSS folder for signal integrity.")
    p.add_argument("--pilot-dir", type=Path, required=True, help="Path to pilot output directory")
    args = p.parse_args()

    pilot_dir: Path = args.pilot_dir
    if not pilot_dir.is_dir():
        print(f"error: not a directory: {pilot_dir}", file=sys.stderr)
        sys.exit(2)

    ok, errors = validate(pilot_dir)
    if ok:
        print("OK: pilot outputs passed validation.")
        sys.exit(0)

    print("FAIL: pilot outputs failed validation:", file=sys.stderr)
    for e in errors:
        print(f"- {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()

