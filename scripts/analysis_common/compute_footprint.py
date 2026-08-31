#!/usr/bin/env python3
"""Reconstruct active compute time for both phases from episode timestamps.

Section 8.6 reports the study's own compute footprint. The figures are not
budgeted but reconstructed here: episodes are ordered by timestamp and the gaps
between consecutive episodes are summed, excluding any gap longer than the idle
threshold so that pauses between sessions do not count as compute.

Read-only. Deterministic.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/results/compute_footprint.json"
IDLE_THRESHOLD_S = 1200  # 20 minutes; gaps above this are treated as pauses


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _canonical_sources() -> dict[str, Path]:
    """Map each canonical episode id to the run directory it was taken from.

    The manifest is authoritative here. Tower of Hanoi comes from the 2026-07-22
    run and TextWorld from the 2026-07-24 re-collection, so an episode id alone
    does not identify a file: the same id exists in the superseded run too.
    """
    manifest = json.loads(
        (ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json").read_text()
    )
    return {e["episode_id"]: Path(e["source_dir"]) for e in manifest["entries"]}


def _stamps_phase1() -> list[dt.datetime]:
    out = []
    for episode_id, source_dir in _canonical_sources().items():
        f = source_dir / f"trace_{episode_id}.jsonl"
        if not f.exists():
            continue
        try:
            first = json.loads(f.open().readline())
        except Exception:
            continue
        if first.get("timestamp_utc"):
            out.append(_parse(first["timestamp_utc"]))
    return out


def _stamps_phase2() -> list[dt.datetime]:
    out = []
    for f in (ROOT / "data/results/phase2").rglob("ep_*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        ts = d.get("timestamp_utc") or d.get("started_at") or d.get("finished_at")
        if ts:
            out.append(_parse(ts))
    return out


def summarise(stamps: list[dt.datetime]) -> dict:
    stamps = sorted(stamps)
    if len(stamps) < 2:
        return {"n_episodes": len(stamps)}
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    active = sum(g for g in gaps if g <= IDLE_THRESHOLD_S)
    return {
        "n_episodes": len(stamps),
        "first": stamps[0].isoformat(),
        "last": stamps[-1].isoformat(),
        "wall_clock_hours": round((stamps[-1] - stamps[0]).total_seconds() / 3600, 1),
        "active_hours": round(active / 3600, 1),
        "idle_threshold_s": IDLE_THRESHOLD_S,
        "episodes_per_active_hour": round(len(stamps) / (active / 3600), 1) if active else None,
    }


def _stamps_discarded_textworld() -> list[dt.datetime]:
    """The TextWorld half of the 2026-07-22 run, discarded and re-collected.

    Real compute was spent on it. Whether it belongs in the reported footprint
    depends on whether the figure answers "what did the analysed data cost" or
    "what did this study cost to run"; both are emitted so either can be cited.
    """
    src = ROOT / "data/results/phase1/phase1_20260722_091125"
    out = []
    for f in src.glob("trace_ep_textworld_*.jsonl"):
        try:
            first = json.loads(f.open().readline())
        except Exception:
            continue
        if first.get("timestamp_utc"):
            out.append(_parse(first["timestamp_utc"]))
    return out


def main() -> int:
    canonical = _stamps_phase1()
    discarded = _stamps_discarded_textworld()
    result = {
        "phase1_canonical_only": summarise(canonical),
        "phase1_including_discarded_run": summarise(canonical + discarded),
        "discarded_textworld_run": summarise(discarded),
        "phase2": summarise(_stamps_phase2()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    tot = (result["phase1_including_discarded_run"]["active_hours"]
           + result["phase2"]["active_hours"])
    result["study_total_active_hours"] = round(tot, 1)
    for phase, d in result.items():
        if not isinstance(d, dict):
            continue
        if d.get("active_hours") is None:
            print(f"{phase}: {d['n_episodes']} episodes, insufficient timestamps")
            continue
        print(f"{phase}: {d['n_episodes']} episodes, {d['active_hours']} active hours "
              f"({d['wall_clock_hours']} h wall clock), "
              f"{d['episodes_per_active_hour']} episodes/h")
    print(f"\nstudy total (all compute actually spent): "
          f"{result['study_total_active_hours']} active hours")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
