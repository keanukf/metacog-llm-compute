"""
Dataset loaders for post-hoc analysis.

This module converts a run folder (pilot / phase1 / phase2) into:
- an episode-level table (one row per `ep_*.json`)
- a step-level table (one row per environment step)

Design goals:
- Works with compact episode JSONs (no `steps_detail` stored) by synthesizing minimal step rows.
- Provides a consistent step-level correctness label for calibration:
  primary policy is `optimal_only` (optimal=1, legal/illegal=0).
- Detects presence of optional sidecar artifacts (logprobs / vc) without requiring them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

CorrectnessPolicy = Literal["optimal_only", "legal_or_optimal"]


@dataclass(frozen=True)
class RunDataset:
    run_dir: Path
    episodes: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    run_metadata: dict[str, Any] | None
    run_info: dict[str, Any] | None
    run_summary: dict[str, Any] | None
    errors: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    return out


def _episode_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("ep_*.json"))


def _sidecar_paths(run_dir: Path, episode_id: str) -> dict[str, str | None]:
    logprobs_json = run_dir / "logprobs" / f"{episode_id}_logprobs.json"
    logprobs_csv = run_dir / "logprobs" / f"{episode_id}_logprobs.csv"
    vc_json = run_dir / "vc" / f"{episode_id}_vc.json"
    vc_csv = run_dir / "vc" / f"{episode_id}_vc.csv"
    trace_jsonl = run_dir / f"trace_{episode_id}.jsonl"
    return {
        "logprobs_json_path": str(logprobs_json) if logprobs_json.exists() else None,
        "logprobs_csv_path": str(logprobs_csv) if logprobs_csv.exists() else None,
        "vc_json_path": str(vc_json) if vc_json.exists() else None,
        "vc_csv_path": str(vc_csv) if vc_csv.exists() else None,
        "trace_jsonl_path": str(trace_jsonl) if trace_jsonl.exists() else None,
    }


def _step_correct_optimal(corr: Any, policy: CorrectnessPolicy) -> int | None:
    if corr is None:
        return None
    if isinstance(corr, str):
        c = corr.strip().lower()
        if policy == "optimal_only":
            if c == "optimal":
                return 1
            if c in {"legal", "illegal"}:
                return 0
            return None
        if policy == "legal_or_optimal":
            if c in {"optimal", "legal"}:
                return 1
            if c == "illegal":
                return 0
            return None
        return None
    if isinstance(corr, (int, float)):
        # Some envs may expose numeric correctness. We treat any positive as "correct".
        # For `optimal_only` this is imperfect, but keeps the analysis usable without env-specific mapping.
        return 1 if float(corr) > 0 else 0
    return None


def _index_correctness_by_step(step_correctness: Any) -> dict[int, Any]:
    out: dict[int, Any] = {}
    if not isinstance(step_correctness, list):
        return out
    for d in step_correctness:
        if not isinstance(d, dict):
            continue
        try:
            idx = int(d.get("step_index"))
        except Exception:
            continue
        out[idx] = d.get("correctness")
    return out


def _synthesize_steps_detail_minimal(ep: dict[str, Any]) -> list[dict[str, Any]]:
    steps = int(ep.get("steps") or ep.get("episode_length_steps") or 0)
    tle_list = ep.get("tle_per_step") or []
    vc_list = ep.get("vc_per_step") or []
    stage_list = ep.get("stage_per_step") or []
    fixed_stage = ep.get("compute_stage") or None
    corr_by_idx = _index_correctness_by_step(ep.get("step_correctness") or [])
    out: list[dict[str, Any]] = []
    for i in range(steps):
        compute_stage = (
            str(stage_list[i])
            if i < len(stage_list) and stage_list[i] is not None
            else str(fixed_stage)
            if fixed_stage is not None
            else "C0"
        )
        tle = tle_list[i] if i < len(tle_list) else None
        vc = vc_list[i] if i < len(vc_list) else None
        out.append(
            {
                "step_index": i,
                "compute_stage": compute_stage,
                "action": "",
                "tokens_generated": 0,
                "lm_calls_this_step": 1,
                "step_wall_time_s": 0.0,
                "tle": tle,
                "vc": vc,
                "correctness": corr_by_idx.get(i),
                "observation_length_chars": 0,
            }
        )
    return out


def _ensure_steps_detail(
    ep: dict[str, Any], *, had_steps_detail: bool
) -> tuple[list[dict[str, Any]], bool]:
    if had_steps_detail and isinstance(ep.get("steps_detail"), list):
        return list(ep["steps_detail"]), False
    return _synthesize_steps_detail_minimal(ep), True


def load_run_dataset(
    run_dir: str | Path,
    *,
    correctness_policy: CorrectnessPolicy = "optimal_only",
) -> RunDataset:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run_dir is not a directory: {run_dir}")

    run_metadata = _read_json(run_dir / "run_metadata.json")
    run_info = _read_json(run_dir / "run_info.json")
    run_summary = _read_json(run_dir / "run_summary.json")
    errors = _read_jsonl(run_dir / "errors.jsonl") if (run_dir / "errors.jsonl").exists() else []

    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    for p in _episode_paths(run_dir):
        raw = _read_json(p)
        if raw is None:
            continue
        ep = dict(raw)
        ep.setdefault("episode_id", p.stem)
        ep_id = str(ep.get("episode_id") or p.stem)

        had_steps_detail = isinstance(raw.get("steps_detail"), list)
        steps_detail, synthesized = _ensure_steps_detail(ep, had_steps_detail=had_steps_detail)
        ep["steps_detail"] = steps_detail
        ep["_steps_detail_synthesized"] = bool(synthesized)

        ep.update(_sidecar_paths(run_dir, ep_id))
        episodes.append(ep)

        ep_len = len(steps_detail)
        denom = float(ep_len) if ep_len > 0 else 1.0
        for sd in steps_detail:
            if not isinstance(sd, dict):
                continue
            row: dict[str, Any] = {}
            # Episode-level join (minimal set; add more downstream if needed)
            for k in (
                "episode_id",
                "domain",
                "instance",
                "compute_stage",
                "strategy",
                "run",
                "task_success",
                "episode_length_steps",
                "steps",
                "total_lm_calls",
                "total_tokens_generated",
                "normalized_compute_cost",
                "efficiency_score",
                "timestamp_utc",
                "wall_clock_time",
                "_steps_detail_synthesized",
            ):
                row[k] = ep.get(k)

            # Step-level fields
            row.update(sd)
            try:
                step_index = int(sd.get("step_index"))
            except Exception:
                continue
            row["step_index"] = step_index
            row["relative_step_position"] = float(step_index) / denom

            # Correctness label (primary)
            corr_raw = sd.get("correctness")
            row["step_correctness_raw"] = corr_raw
            row["step_correct_optimal"] = _step_correct_optimal(corr_raw, correctness_policy)

            # Unnest TLE convenience fields
            tle = sd.get("tle")
            if isinstance(tle, dict):
                row["tle_mean_entropy"] = tle.get("mean_entropy")
                row["tle_max_entropy"] = tle.get("max_entropy")
            else:
                row["tle_mean_entropy"] = None
                row["tle_max_entropy"] = None

            # Sidecar paths
            for sk in (
                "logprobs_json_path",
                "logprobs_csv_path",
                "vc_json_path",
                "vc_csv_path",
                "trace_jsonl_path",
            ):
                row[sk] = ep.get(sk)

            steps.append(row)

    return RunDataset(
        run_dir=run_dir,
        episodes=episodes,
        steps=steps,
        run_metadata=run_metadata,
        run_info=run_info,
        run_summary=run_summary,
        errors=errors,
    )


def episodes_frame(dataset: RunDataset):
    """
    Return episodes as a pandas DataFrame if pandas is available, else a list[dict].
    """
    try:
        import pandas as pd  # type: ignore

        return pd.DataFrame(dataset.episodes)
    except Exception:
        return dataset.episodes


def steps_frame(dataset: RunDataset):
    """
    Return steps as a pandas DataFrame if pandas is available, else a list[dict].
    """
    try:
        import pandas as pd  # type: ignore

        return pd.DataFrame(dataset.steps)
    except Exception:
        return dataset.steps


def validate_analysis_schema(
    steps_rows: Iterable[dict[str, Any]],
    *,
    require_columns: tuple[str, ...] = (
        "episode_id",
        "domain",
        "instance",
        "step_index",
        "relative_step_position",
        "step_correct_optimal",
        "tle_mean_entropy",
        "vc",
        "compute_stage",
        "strategy",
    ),
) -> dict[str, Any]:
    """
    Lightweight schema check for step rows.

    Returns a dict with `missing_columns` and simple counts useful for run health panels.
    """
    missing: set[str] = set()
    n = 0
    n_missing_vc = 0
    n_missing_tle = 0
    n_missing_label = 0
    n_synth = 0
    for r in steps_rows:
        if not isinstance(r, dict):
            continue
        n += 1
        for c in require_columns:
            if c not in r:
                missing.add(c)
        if r.get("vc") is None:
            n_missing_vc += 1
        if r.get("tle_mean_entropy") is None:
            n_missing_tle += 1
        if r.get("step_correct_optimal") is None:
            n_missing_label += 1
        if bool(r.get("_steps_detail_synthesized")):
            n_synth += 1
    return {
        "n_steps": int(n),
        "missing_columns": sorted(missing),
        "missing_vc_rate": (n_missing_vc / n) if n else 0.0,
        "missing_tle_rate": (n_missing_tle / n) if n else 0.0,
        "missing_label_rate": (n_missing_label / n) if n else 0.0,
        "synthesized_steps_rate": (n_synth / n) if n else 0.0,
    }
