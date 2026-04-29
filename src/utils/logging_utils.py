"""
Structured JSON logging for experiment episodes.
One JSON file per episode; used for pilot_calibration and phase1/phase2 results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Omitted from main episode JSON when ``compact=True`` — full detail lives in logprob / VC sidecars.
_EPISODE_STORAGE_DETAIL_KEYS = frozenset({"steps_detail", "vc_detail_per_step", "logprob_raw_per_step"})


def compact_episode_for_storage(data: dict[str, Any]) -> dict[str, Any]:
    """
    Drop per-step verbose fields meant for sidecar files only.

    Keeps summary vectors like ``tle_per_step``, ``vc_per_step``, ``step_correctness``.
    """
    return {k: v for k, v in data.items() if k not in _EPISODE_STORAGE_DETAIL_KEYS}


def log_episode(
    episode_id: str,
    data: dict[str, Any],
    path: str | Path,
    tracker: Any = None,
    *,
    compact: bool = True,
) -> Path:
    """
    Write a single episode's data as one JSON file.

    Args:
        episode_id: Unique id (e.g. ep_{domain}_{instance}_{stage}_{run}).
        data: Dict with keys such as TLE, VC, task_success, steps, lm_calls (env steps, legacy),
        total_lm_calls, tokens, wall_clock_time.
        path: Directory or full file path; if directory, file is path / f"{episode_id}.json".
        tracker: Reserved for optional hooks; ignored by this function (file write only).
        compact: If True (default), omit ``steps_detail``, ``vc_detail_per_step``, and
            ``logprob_raw_per_step`` — use sidecar JSONs under ``logprobs/`` / ``vc/`` for those.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    if path.suffix != ".json":
        path = path / f"{episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = compact_episode_for_storage(data) if compact else dict(data)
    with open(path, "w") as f:
        json.dump(to_write, f, indent=2)
    return path


def write_step_trace_line(trace_path: str | Path, record: dict[str, Any]) -> None:
    """
    Append one JSON object as a line to a per-episode trace file (JSONL).

    Used for full prompt/response observability alongside compact episode JSON.
    """
    p = Path(trace_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class EpisodeLogger:
    """Logger that writes one JSON file per episode to a results directory."""

    def __init__(self, results_dir: str | Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def log(self, episode_id: str, data: dict[str, Any]) -> Path:
        """Log one episode; returns path to written file."""
        return log_episode(episode_id, data, self.results_dir)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _try_git_commit(repo_root: str | Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def _try_gpu_info() -> tuple[str | None, float | None]:
    """
    Returns (gpu_name, vram_total_gb) if torch+CUDA are available, else (None, None).
    """
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return None, None
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = float(props.total_memory) / (1024**3)
        return name, vram_total_gb
    except Exception:
        return None, None


def write_run_metadata(
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    *,
    script: str,
    config_path: str | Path,
    pilot_mode: str,
    model_name: str,
    model_dtype: str,
    domains: list[str],
    total_episodes_planned: int,
    resumed_from: int,
    repo_root: str | Path | None = None,
) -> Path:
    """
    Write `run_metadata.json` into checkpoint_dir.

    This file is meant to make overnight runs reproducible and debuggable.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    meta = {
        "run_id": str(uuid.uuid4()),
        "script": str(script),
        "config_path": str(config_path),
        "config_hash": _sha256_file(config_path),
        "model_name": str(model_name),
        "model_dtype": str(model_dtype),
        "pilot_mode": str(pilot_mode),
        "git_commit": _try_git_commit(repo_root),
        "timestamp_start_utc": _iso_utc_now(),
        "python_version": sys.version.split()[0],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_name": None,
        "vram_total_gb": None,
        "domains": list(domains),
        "total_episodes_planned": int(total_episodes_planned),
        "resumed_from": int(resumed_from),
    }
    gpu_name, vram_total = _try_gpu_info()
    meta["gpu_name"] = gpu_name
    meta["vram_total_gb"] = vram_total

    path = checkpoint_dir / "run_metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def _synthesize_steps_detail(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build a minimal `steps_detail` from legacy episode keys when missing.
    """
    if isinstance(episode.get("steps_detail"), list):
        return list(episode["steps_detail"])
    steps = int(episode.get("steps") or episode.get("episode_length_steps") or 0)
    tle_list = episode.get("tle_per_step") or []
    vc_list = episode.get("vc_per_step") or []
    stage_list = episode.get("stage_per_step") or []
    fixed_stage = episode.get("compute_stage") or None
    step_correctness = episode.get("step_correctness") or []
    corr_by_idx: dict[int, Any] = {}
    if isinstance(step_correctness, list):
        for d in step_correctness:
            if not isinstance(d, dict):
                continue
            try:
                idx = int(d.get("step_index"))
            except Exception:
                continue
            corr_by_idx[idx] = d.get("correctness")

    out: list[dict[str, Any]] = []
    for i in range(steps):
        compute_stage = (
            str(stage_list[i]) if i < len(stage_list) and stage_list[i] is not None else str(fixed_stage) if fixed_stage is not None else "C0"
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


def load_episodes(
    checkpoint_dir: str | Path,
    as_dataframe: bool = False,
) -> list[dict[str, Any]] | Any:
    """
    Load all episode JSONs from a directory.

    Backward compatibility:
    - Episodes without `steps_detail` will get a synthesized minimal `steps_detail`.

    Args:
        checkpoint_dir: Directory containing `ep_*.json`.
        as_dataframe: If True, return a pandas DataFrame (requires pandas).
    """
    checkpoint_dir = Path(checkpoint_dir)
    episodes: list[dict[str, Any]] = []
    for p in sorted(checkpoint_dir.glob("ep_*.json")):
        try:
            with open(p) as f:
                ep = json.load(f)
            if isinstance(ep, dict):
                ep = dict(ep)
                ep["steps_detail"] = _synthesize_steps_detail(ep)
                episodes.append(ep)
        except Exception:
            continue

    if not as_dataframe:
        return episodes

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError("pandas is required for as_dataframe=True") from e
    return pd.DataFrame(episodes)


def load_steps(checkpoint_dir: str | Path):
    """
    Load all episodes and flatten `steps_detail` into one row per step.

    By default this returns a pandas DataFrame when pandas is importable. If pandas is not
    available in the runtime environment, it returns a plain `list[dict]` instead so that
    analysis code can still run in minimal setups.

    Returns:
        pandas.DataFrame (preferred) or list[dict] fallback with episode-level columns joined.
    """
    episodes = load_episodes(checkpoint_dir, as_dataframe=False)
    rows: list[dict[str, Any]] = []
    episode_cols = [
        "episode_id",
        "domain",
        "instance",
        "compute_stage",
        "strategy",
        "run",
        "task_success",
        "episode_length_steps",
        "total_lm_calls",
        "total_tokens_generated",
        "normalized_compute_cost",
        "efficiency_score",
        "timestamp_utc",
    ]
    for ep in episodes:
        base = {k: ep.get(k) for k in episode_cols if k in ep}
        for sd in ep.get("steps_detail") or []:
            if not isinstance(sd, dict):
                continue
            row = dict(base)
            row.update(sd)
            # Unnest tle fields for convenience
            tle = sd.get("tle")
            if isinstance(tle, dict):
                row["tle_mean_entropy"] = tle.get("mean_entropy")
                row["tle_max_entropy"] = tle.get("max_entropy")
            rows.append(row)
    try:
        import pandas as pd  # type: ignore

        return pd.DataFrame(rows)
    except Exception:
        return rows


def write_logprob_distribution_artifacts(
    episode_id: str,
    logprob_raw_per_step: list[Any] | None,
    output_dir: str | Path,
    *,
    export_format: str = "json",
    logprob_subdir: str = "logprobs",
) -> list[Path]:
    """
    Write optional sidecar files with per-env-step completion token logprob rows (top-k optional).

    ``export_format``: ``json``, ``csv``, or ``both``. JSON is the canonical nested structure;
    CSV is a long tidy table for pandas (one row per top-k rank at each completion token).
    """
    from src.signals.token_entropy import softmax_probs_from_top_logprobs

    if not logprob_raw_per_step:
        return []
    out_dir = Path(output_dir) / logprob_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{episode_id}_logprobs"
    any_multi = False
    steps_payload: list[dict[str, Any]] = []
    for i, lp in enumerate(logprob_raw_per_step):
        # Schema v1 per step: lp is list[dict] (token-level records).
        if isinstance(lp, list) and (not lp or isinstance(lp[0], dict)):
            steps_payload.append({"step_index": i, "logprob_tokens": lp if lp is not None else []})
            continue
        # Schema v2 per step (C2): lp is list[list[dict] | None] with one entry per sample.
        if isinstance(lp, list) and lp and (lp[0] is None or isinstance(lp[0], list)):
            any_multi = True
            samples_payload = []
            for si, s_lp in enumerate(lp):
                samples_payload.append(
                    {
                        "sample_index": int(si),
                        "logprob_tokens": s_lp if isinstance(s_lp, list) else [],
                    }
                )
            steps_payload.append({"step_index": i, "samples": samples_payload})
            continue
        # Unknown / disabled for this step
        steps_payload.append({"step_index": i, "logprob_tokens": []})
    body: dict[str, Any] = {
        "episode_id": episode_id,
        "schema_version": 2 if any_multi else 1,
        "description": (
            "Per env step: token-level logprob records. "
            "Schema v1 uses steps[].logprob_tokens; schema v2 may use steps[].samples[].logprob_tokens for multi-sample stages."
        ),
        "steps": steps_payload,
    }
    written: list[Path] = []
    fmt = export_format.strip().lower()
    if fmt in ("json", "both"):
        p = out_dir / f"{stem}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2)
        written.append(p)
    if fmt in ("csv", "both"):
        p_csv = out_dir / f"{stem}.csv"
        with open(p_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "episode_id",
                    "env_step_index",
                    "sample_index",
                    "completion_token_index",
                    "rank_in_topk",
                    "token",
                    "logprob",
                    "p_renorm_topk",
                ]
            )
            for env_i, lp in enumerate(logprob_raw_per_step):
                # Single-sample: list[dict]
                if isinstance(lp, list) and (not lp or isinstance(lp[0], dict)):
                    lp_list = lp
                    for tok_i, tok in enumerate(lp_list):
                        if not isinstance(tok, dict):
                            continue
                        top = tok.get("top_logprobs")
                        if isinstance(top, list) and top:
                            cands = [x for x in top if isinstance(x, dict) and x.get("logprob") is not None]
                            probs = softmax_probs_from_top_logprobs(top)
                            for rank, (cand, pr) in enumerate(zip(cands, probs)):
                                w.writerow(
                                    [
                                        episode_id,
                                        env_i,
                                        0,
                                        tok_i,
                                        rank,
                                        cand.get("token", ""),
                                        cand.get("logprob", ""),
                                        f"{pr:.8g}",
                                    ]
                                )
                        else:
                            w.writerow(
                                [
                                    episode_id,
                                    env_i,
                                    0,
                                    tok_i,
                                    0,
                                    tok.get("token", ""),
                                    tok.get("logprob", ""),
                                    "1.0",
                                ]
                            )
                    continue
                # Multi-sample: list[list[dict] | None]
                if isinstance(lp, list) and lp and (lp[0] is None or isinstance(lp[0], list)):
                    for si, s_lp in enumerate(lp):
                        if not isinstance(s_lp, list) or not s_lp:
                            continue
                        for tok_i, tok in enumerate(s_lp):
                            if not isinstance(tok, dict):
                                continue
                            top = tok.get("top_logprobs")
                            if isinstance(top, list) and top:
                                cands = [x for x in top if isinstance(x, dict) and x.get("logprob") is not None]
                                probs = softmax_probs_from_top_logprobs(top)
                                for rank, (cand, pr) in enumerate(zip(cands, probs)):
                                    w.writerow(
                                        [
                                            episode_id,
                                            env_i,
                                            si,
                                            tok_i,
                                            rank,
                                            cand.get("token", ""),
                                            cand.get("logprob", ""),
                                            f"{pr:.8g}",
                                        ]
                                    )
                            else:
                                w.writerow(
                                    [
                                        episode_id,
                                        env_i,
                                        si,
                                        tok_i,
                                        0,
                                        tok.get("token", ""),
                                        tok.get("logprob", ""),
                                        "1.0",
                                    ]
                                )
        written.append(p_csv)
    return written


def write_vc_distribution_artifacts(
    episode_id: str,
    vc_detail_per_step: list[dict[str, Any] | None] | None,
    output_dir: str | Path,
    *,
    export_format: str = "json",
    vc_subdir: str = "vc",
) -> list[Path]:
    """
    Write optional sidecar JSON/CSV for verbalized-confidence follow-up calls (per env step).

    Each step may contain ``vc_prompt``, ``vc_raw_text``, ``vc_value``, ``vc_logprobs`` (per-token rows).
    Default ``vc_subdir`` is ``vc`` (parallel to ``logprobs/`` for TLE sidecars).
    """
    from src.signals.token_entropy import softmax_probs_from_top_logprobs

    if not vc_detail_per_step:
        return []
    out_dir = Path(output_dir) / str(vc_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{episode_id}_vc"
    steps_payload: list[dict[str, Any]] = []
    for i, d in enumerate(vc_detail_per_step):
        if d is None:
            steps_payload.append({"step_index": i, "vc_record": None})
        else:
            steps_payload.append({"step_index": i, "vc_record": dict(d)})
    body: dict[str, Any] = {
        "episode_id": episode_id,
        "schema_version": 1,
        "description": "Per env step: VC follow-up metadata and optional per-token vc_logprobs.",
        "steps": steps_payload,
    }
    written: list[Path] = []
    fmt = export_format.strip().lower()
    if fmt in ("json", "both"):
        p = out_dir / f"{stem}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2)
        written.append(p)
    if fmt in ("csv", "both"):
        p_csv = out_dir / f"{stem}.csv"
        with open(p_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "episode_id",
                    "env_step_index",
                    "vc_token_index",
                    "rank_in_topk",
                    "token",
                    "logprob",
                    "p_renorm_topk",
                ]
            )
            for env_i, d in enumerate(vc_detail_per_step):
                if not isinstance(d, dict):
                    continue
                lp_list = d.get("vc_logprobs")
                if not isinstance(lp_list, list) or not lp_list:
                    continue
                for tok_i, tok in enumerate(lp_list):
                    if not isinstance(tok, dict):
                        continue
                    top = tok.get("top_logprobs")
                    if isinstance(top, list) and top:
                        cands = [
                            x
                            for x in top
                            if isinstance(x, dict) and x.get("logprob") is not None
                        ]
                        probs = softmax_probs_from_top_logprobs(top)
                        for rank, (cand, pr) in enumerate(zip(cands, probs)):
                            w.writerow(
                                [
                                    episode_id,
                                    env_i,
                                    tok_i,
                                    rank,
                                    cand.get("token", ""),
                                    cand.get("logprob", ""),
                                    f"{pr:.8g}",
                                ]
                            )
                    else:
                        w.writerow(
                            [
                                episode_id,
                                env_i,
                                tok_i,
                                0,
                                tok.get("token", ""),
                                tok.get("logprob", ""),
                                "1.0",
                            ]
                        )
        written.append(p_csv)
    return written
