"""
Compact JSON debug views derived from per-step ``trace_*.jsonl`` files.

Human-readable pipeline summaries (truncated prompts/responses) for C0/C1/C2 runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HEAD_CHARS = 800
DEFAULT_TAIL_CHARS = 800
_HISTORY_TAIL_ENTRIES = 3


def truncate_text(
    text: str | None,
    *,
    head_chars: int = DEFAULT_HEAD_CHARS,
    tail_chars: int = DEFAULT_TAIL_CHARS,
) -> dict[str, Any] | None:
    """Return head/tail slice metadata for long strings; ``None`` when empty."""
    if not text:
        return None
    s = str(text)
    n = len(s)
    if n == 0:
        return None
    h = max(0, int(head_chars))
    t = max(0, int(tail_chars))
    if n <= h + t + 20:
        return {
            "head": s,
            "tail": "",
            "length": n,
            "sha256_prefix": hashlib.sha256(s.encode("utf-8")).hexdigest()[:16],
            "truncated": False,
        }
    return {
        "head": s[:h],
        "tail": s[-t:] if t else "",
        "length": n,
        "sha256_prefix": hashlib.sha256(s.encode("utf-8")).hexdigest()[:16],
        "truncated": True,
    }


def _trunc(
    text: str | None,
    *,
    head_chars: int,
    tail_chars: int,
) -> dict[str, Any] | None:
    return truncate_text(text, head_chars=head_chars, tail_chars=tail_chars)


def _compact_io(
    prompt: str | None,
    response: str | None,
    *,
    head_chars: int,
    tail_chars: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prompt": _trunc(prompt, head_chars=head_chars, tail_chars=tail_chars),
        "response": _trunc(response, head_chars=head_chars, tail_chars=tail_chars),
    }
    if extra:
        out.update(extra)
    return out


def _gen_fields(sc: dict[str, Any]) -> dict[str, Any]:
    gen: dict[str, Any] = {}
    for key in (
        "temperature",
        "max_tokens",
        "stop",
        "tokens_generated",
        "sample_index",
        "is_winner",
        "raw_first_line",
        "action_exec",
        "action_normalized",
        "mean_logprob",
        "tle",
    ):
        if key in sc and sc[key] is not None:
            gen[key] = sc[key]
    return gen


def compact_subcall(
    sc: dict[str, Any],
    *,
    head_chars: int = DEFAULT_HEAD_CHARS,
    tail_chars: int = DEFAULT_TAIL_CHARS,
) -> dict[str, Any]:
    """Normalize one subcall record (cot / verify / sample / generic)."""
    kind = str(sc.get("kind") or "unknown").strip().lower()
    base = _compact_io(
        sc.get("prompt"),
        sc.get("response"),
        head_chars=head_chars,
        tail_chars=tail_chars,
        extra={"kind": kind},
    )
    gen = _gen_fields(sc)
    if gen:
        base["gen"] = gen
    for key in ("fallback_source", "raw_first_line", "action_exec", "action_normalized"):
        if sc.get(key) not in (None, ""):
            base[key] = sc[key]
    return base


def _subcall_by_kind(subcalls: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for sc in subcalls:
        if str(sc.get("kind") or "").strip().lower() == kind:
            return sc
    return None


def _history_tail(history: Any) -> list[str] | None:
    if not isinstance(history, list) or not history:
        return None
    tail = [str(x) for x in history[-_HISTORY_TAIL_ENTRIES:]]
    return tail


def compact_step_from_trace(
    row: dict[str, Any],
    *,
    head_chars: int = DEFAULT_HEAD_CHARS,
    tail_chars: int = DEFAULT_TAIL_CHARS,
) -> dict[str, Any]:
    """Build one debug-view step object from a trace JSONL row."""
    stage = str(row.get("compute_stage") or "")
    _call_detail_raw = row.get("call_detail")
    call_detail: dict[str, Any] = _call_detail_raw if isinstance(_call_detail_raw, dict) else {}
    subcalls_raw = call_detail.get("subcalls")
    subcalls: list[dict[str, Any]] = (
        [sc for sc in subcalls_raw if isinstance(sc, dict)]
        if isinstance(subcalls_raw, list)
        else []
    )

    pipeline: dict[str, Any] = {}

    pipeline["primary"] = _compact_io(
        row.get("prompt_full"),
        row.get("response_full"),
        head_chars=head_chars,
        tail_chars=tail_chars,
    )

    cot_sc = _subcall_by_kind(subcalls, "cot")
    cot_prompt = row.get("cot_prompt") or (cot_sc.get("prompt") if cot_sc else None)
    cot_response = row.get("cot_response") or (cot_sc.get("response") if cot_sc else None)
    if cot_prompt or cot_response:
        cot_block = _compact_io(
            cot_prompt,
            cot_response,
            head_chars=head_chars,
            tail_chars=tail_chars,
        )
        if cot_sc:
            gen = _gen_fields(cot_sc)
            if gen:
                cot_block["gen"] = gen
        pipeline["cot"] = cot_block

    verify_sc = _subcall_by_kind(subcalls, "verify")
    verify_prompt = row.get("verify_prompt") or (verify_sc.get("prompt") if verify_sc else None)
    verify_response = row.get("verify_response") or (
        verify_sc.get("response") if verify_sc else None
    )
    if verify_prompt or verify_response:
        verify_block = _compact_io(
            verify_prompt,
            verify_response,
            head_chars=head_chars,
            tail_chars=tail_chars,
        )
        parse_info: dict[str, Any] = {}
        for key in (
            "parse_method",
            "draft_action",
            "draft_status",
            "fallback_source",
            "draft_reasoning_raw",
        ):
            val = (
                call_detail.get(key)
                if key in call_detail
                else verify_sc.get(key)
                if verify_sc
                else None
            )
            if val not in (None, ""):
                parse_info[key] = val
        if parse_info:
            verify_block["parse"] = parse_info
        if verify_sc:
            gen = _gen_fields(verify_sc)
            if gen:
                verify_block["gen"] = gen
        pipeline["verify"] = verify_block

    vc_prompt = row.get("vc_followup_prompt")
    vc_response = row.get("vc_followup_response")
    if vc_prompt or vc_response:
        pipeline["vc_followup"] = _compact_io(
            vc_prompt,
            vc_response,
            head_chars=head_chars,
            tail_chars=tail_chars,
        )

    sample_subcalls = [sc for sc in subcalls if str(sc.get("kind") or "").lower() == "sample"]
    if sample_subcalls:
        pipeline["c2_samples"] = [
            compact_subcall(sc, head_chars=head_chars, tail_chars=tail_chars)
            for sc in sample_subcalls
        ]
        vote_summary: dict[str, Any] = {}
        for key in (
            "method",
            "n_samples",
            "winner_index",
            "winning_vote_key",
            "tie_broken",
            "vote_counts",
            "vote_agreement",
            "unique_actions",
            "winner_raw_first_line",
            "winner_mean_logprob",
        ):
            if call_detail.get(key) is not None:
                vote_summary[key] = call_detail[key]
        if vote_summary:
            pipeline["c2_vote"] = vote_summary

    pipeline["final"] = {
        "action_parsed": row.get("action_parsed"),
        "observation_before": _trunc(
            row.get("observation_before"),
            head_chars=head_chars,
            tail_chars=tail_chars,
        ),
        "observation_after": _trunc(
            row.get("observation_after"),
            head_chars=head_chars,
            tail_chars=tail_chars,
        ),
        "history_tail": _history_tail(row.get("history_snapshot")),
    }

    signals: dict[str, Any] = {}
    if row.get("tle") is not None:
        signals["tle"] = row.get("tle")
    if row.get("vc") is not None:
        signals["vc"] = row.get("vc")

    return {
        "step_index": row.get("step_index"),
        "compute_stage": stage,
        "action_executed": row.get("action_parsed"),
        "correctness": row.get("correctness"),
        "signals": signals or None,
        "timing": {
            "wall_s": row.get("step_wall_time_s"),
            "lm_calls": row.get("lm_calls"),
            "tokens_generated": row.get("tokens_generated"),
        },
        "pipeline": pipeline,
    }


def _episode_id_from_trace_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("trace_"):
        return stem[len("trace_") :]
    return stem


def _load_episode_metadata(run_dir: Path, episode_id: str) -> dict[str, Any] | None:
    candidates = [
        run_dir / f"{episode_id}.json",
        run_dir / "episodes" / f"{episode_id}.json",
    ]
    for p in candidates:
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                continue
    for p in sorted(run_dir.glob("*.json")):
        if p.name in ("run_info.json", "run_summary.json") or p.parent.name == "debug_views":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and str(data.get("episode_id") or "") == episode_id:
            return data
    return None


def _episode_summary_from_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {}
    keys = (
        "episode_id",
        "domain",
        "compute_stage",
        "strategy",
        "task_success",
        "steps",
        "total_lm_calls",
        "wall_clock_time",
        "instance",
        "run",
    )
    return {k: meta[k] for k in keys if k in meta}


def _read_trace_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read trace %s: %s", path, e)
        return rows
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid JSON at %s:%d", path, line_no)
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def build_run_debug_views(
    run_dir: str | Path,
    *,
    head_chars: int = DEFAULT_HEAD_CHARS,
    tail_chars: int = DEFAULT_TAIL_CHARS,
) -> dict[str, Any] | None:
    """
    Read all ``trace_*.jsonl`` under ``run_dir`` and write ``debug_views/`` artifacts.

    Returns the run summary dict, or ``None`` if no trace files were found.
    """
    run_dir = Path(run_dir)
    trace_files = sorted(run_dir.glob("trace_*.jsonl"))
    if not trace_files:
        logger.info("No trace_*.jsonl in %s — skipping debug_views", run_dir)
        return None

    debug_dir = run_dir / "debug_views"
    debug_dir.mkdir(parents=True, exist_ok=True)

    parse_methods: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    total_steps = 0
    empty_actions = 0
    episodes_out: list[dict[str, Any]] = []

    for trace_path in trace_files:
        episode_id = _episode_id_from_trace_path(trace_path)
        rows = _read_trace_lines(trace_path)
        meta = _load_episode_metadata(run_dir, episode_id)
        ep_summary = _episode_summary_from_metadata(meta)

        steps: list[dict[str, Any]] = []
        ep_empty = 0
        ep_parse: Counter[str] = Counter()
        for row in rows:
            step = compact_step_from_trace(row, head_chars=head_chars, tail_chars=tail_chars)
            steps.append(step)
            total_steps += 1
            stage = str(step.get("compute_stage") or "")
            if stage:
                stages[stage] += 1
            action = step.get("action_executed")
            if not action or not str(action).strip():
                empty_actions += 1
                ep_empty += 1
            verify = (step.get("pipeline") or {}).get("verify") or {}
            parse_block = verify.get("parse") or {}
            method = parse_block.get("parse_method")
            if method:
                parse_methods[str(method)] += 1
                ep_parse[str(method)] += 1

        ep_doc: dict[str, Any] = {
            "episode_id": episode_id,
            "trace_file": str(trace_path.relative_to(run_dir)),
            "episode_summary": ep_summary,
            "step_count": len(steps),
            "empty_actions": ep_empty,
            "parse_method_counts": dict(ep_parse),
            "steps": steps,
        }
        if meta and meta.get("compute_stage") and not ep_summary.get("compute_stage"):
            ep_doc["compute_stage"] = meta.get("compute_stage")

        ep_path = debug_dir / f"episode_{episode_id}.json"
        with open(ep_path, "w", encoding="utf-8") as f:
            json.dump(ep_doc, f, indent=2, ensure_ascii=False)

        episodes_out.append(
            {
                "episode_id": episode_id,
                "trace_file": ep_doc["trace_file"],
                "debug_file": f"debug_views/episode_{episode_id}.json",
                "step_count": len(steps),
                "empty_actions": ep_empty,
                "compute_stage": ep_summary.get("compute_stage") or ep_doc.get("compute_stage"),
            }
        )

    run_id = run_dir.name
    run_info_path = run_dir / "run_info.json"
    if run_info_path.is_file():
        try:
            ri = json.loads(run_info_path.read_text(encoding="utf-8"))
            if isinstance(ri, dict) and ri.get("timestamp_utc"):
                run_id = f"{run_dir.name}"
        except (json.JSONDecodeError, OSError):
            pass

    summary: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "trace_files_found": len(trace_files),
        "episodes_built": len(episodes_out),
        "total_steps": total_steps,
        "empty_actions": empty_actions,
        "parse_method_histogram": dict(parse_methods),
        "compute_stage_histogram": dict(stages),
        "head_chars": int(head_chars),
        "tail_chars": int(tail_chars),
        "episodes": episodes_out,
    }

    summary_path = debug_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def resolve_step_trace_flags(config: dict) -> tuple[bool, bool, int, int]:
    """
    Return ``(save_step_traces, write_debug_views, head_chars, tail_chars)``.

    When debug views are enabled, step traces are forced on (debug views need trace JSONL).
    """
    lg = config.get("logging") or {}
    write_debug = bool(lg.get("write_debug_views", True))
    save_traces = bool(lg.get("save_step_traces", False))
    head = int(lg.get("debug_view_head_chars", DEFAULT_HEAD_CHARS))
    tail = int(lg.get("debug_view_tail_chars", DEFAULT_TAIL_CHARS))
    if write_debug and not save_traces:
        logger.warning(
            "logging.write_debug_views is true but save_step_traces is false; "
            "enabling save_step_traces for this run."
        )
        save_traces = True
    return save_traces, write_debug, head, tail
