"""Batch- and temperature-invariance probes for execution-layer validation."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.signals.token_entropy import extract_action_tle_from_response
from src.utils.inference.logprob_invariance import (
    probe_temperature_invariance,
    resolve_tle_invariance_eps,
)


@dataclass(frozen=True)
class LoadConstellation:
    constellation_id: str
    pool_size: int
    filler_max_tokens: int


def _action_window_tle(text: str, logprobs: list[dict[str, Any]] | None) -> dict[str, float] | None:
    """TLE at committed-action window (same as agent path)."""
    return extract_action_tle_from_response(text or "", logprobs)


def measure_solo_baselines(
    backend: Any,
    probes: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 32,
) -> dict[str, dict[str, float | None]]:
    """Solo (N=1, no background load) TLE baselines per probe id."""
    baselines: dict[str, dict[str, float | None]] = {}
    for probe in probes:
        pid = str(probe.get("id") or probe.get("prompt", "")[:20])
        prompt = str(probe["prompt"])
        text, lp = backend.generate(
            prompt,
            logprobs=True,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=False,
        )
        tle = _action_window_tle(text, lp)
        baselines[pid] = {
            "mean_entropy": (tle or {}).get("mean_entropy") if tle else None,
            "max_entropy": (tle or {}).get("max_entropy") if tle else None,
        }
    return baselines


def _filler_request(backend: Any, *, max_tokens: int, stop_event: threading.Event) -> None:
    prompt = (
        "You are in a large dungeon with many rooms. Describe exploring room by room "
        "with detailed observations."
    )
    while not stop_event.is_set():
        try:
            backend.generate(
                prompt,
                logprobs=False,
                max_tokens=max_tokens,
                temperature=0.7,
                enable_thinking=False,
            )
        except Exception:
            time.sleep(0.05)


def measure_under_load(
    backend: Any,
    probe: dict[str, str],
    *,
    constellation: LoadConstellation,
    temperature: float = 0.3,
    max_tokens: int = 32,
) -> dict[str, float | None]:
    """Fire probe while background filler threads saturate the server pool."""
    stop_event = threading.Event()
    fillers: list[threading.Thread] = []
    n_fillers = max(0, constellation.pool_size - 1)
    for _ in range(n_fillers):
        t = threading.Thread(
            target=_filler_request,
            args=(backend,),
            kwargs={"max_tokens": constellation.filler_max_tokens, "stop_event": stop_event},
            daemon=True,
        )
        t.start()
        fillers.append(t)
    time.sleep(0.05)
    try:
        text, lp = backend.generate(
            str(probe["prompt"]),
            logprobs=True,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=False,
        )
        tle = _action_window_tle(text, lp)
        return {
            "mean_entropy": (tle or {}).get("mean_entropy") if tle else None,
            "max_entropy": (tle or {}).get("max_entropy") if tle else None,
        }
    finally:
        stop_event.set()
        for t in fillers:
            t.join(timeout=5.0)


def run_batch_invariance_probe(
    backend: Any,
    probes: list[dict[str, str]],
    *,
    max_concurrent_episodes: int,
    eps: float | None = None,
    temperature: float = 0.3,
    max_tokens: int = 32,
) -> dict[str, Any]:
    """
    Batch-invariance probe: |dTLE(solo vs under_load)| over probes and load constellations.

    Primary metric: ``mean_entropy`` at committed-action window.

    The gate is scoped to committed-action-representative probes (``gating`` truthy,
    default ``True``). Probes flagged ``gating: false`` (e.g. underspecified prompts
    that elicit free-form multi-token generation rather than a single committed action)
    are still measured and reported under ``details``/``diagnostic_*`` for transparency,
    but do not drive ``passed`` — they are out of scope for the committed-action TLE
    contract that the experiment actually relies on.
    """
    solo = measure_solo_baselines(backend, probes, temperature=temperature, max_tokens=max_tokens)
    constellations = [
        LoadConstellation("pool2_short", 2, 16),
        LoadConstellation("pool2_long", 2, 64),
        LoadConstellation(
            f"pool{max_concurrent_episodes}_med",
            max_concurrent_episodes,
            32,
        ),
    ]
    if max_concurrent_episodes >= 3:
        constellations.append(
            LoadConstellation(f"pool{max_concurrent_episodes}_long", max_concurrent_episodes, 96)
        )

    gating_ids = {
        str(p.get("id") or p.get("prompt", "")[:20]) for p in probes if p.get("gating", True)
    }

    max_dtle = 0.0
    max_dtle_secondary = 0.0
    worst: dict[str, Any] | None = None
    diagnostic_max_dtle = 0.0
    diagnostic_worst: dict[str, Any] | None = None
    details: list[dict[str, Any]] = []

    for probe in probes:
        pid = str(probe.get("id") or probe.get("prompt", "")[:20])
        is_gating = pid in gating_ids
        base_mean = solo.get(pid, {}).get("mean_entropy")
        base_max = solo.get(pid, {}).get("max_entropy")
        if base_mean is None:
            continue
        for const in constellations:
            under = measure_under_load(
                backend,
                probe,
                constellation=const,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            u_mean = under.get("mean_entropy")
            u_max = under.get("max_entropy")
            d_mean = abs(float(base_mean) - float(u_mean)) if u_mean is not None else None
            d_max = (
                abs(float(base_max) - float(u_max))
                if base_max is not None and u_max is not None
                else None
            )
            row = {
                "probe_id": pid,
                "gating": is_gating,
                "constellation_id": const.constellation_id,
                "pool_size": const.pool_size,
                "filler_max_tokens": const.filler_max_tokens,
                "solo_mean_entropy": base_mean,
                "under_load_mean_entropy": u_mean,
                "dtle_mean": d_mean,
                "dtle_max": d_max,
            }
            details.append(row)
            if d_mean is not None:
                if is_gating:
                    if d_mean > max_dtle:
                        max_dtle = d_mean
                        worst = row
                    if d_max is not None and d_max > max_dtle_secondary:
                        max_dtle_secondary = d_max
                else:
                    if d_mean > diagnostic_max_dtle:
                        diagnostic_max_dtle = d_mean
                        diagnostic_worst = row

    threshold = float(eps) if eps is not None else resolve_tle_invariance_eps([])
    passed = max_dtle <= threshold
    return {
        "passed": passed,
        "max_dtle": max_dtle,
        "max_dtle_secondary": max_dtle_secondary,
        "worst_constellation": worst,
        "gating_probe_ids": sorted(gating_ids),
        "diagnostic_max_dtle": diagnostic_max_dtle,
        "diagnostic_worst_constellation": diagnostic_worst,
        "gate_scope_note": (
            "passed/max_dtle cover committed-action-representative probes only "
            "(gating=true). diagnostic_* covers non-gating probes, reported for "
            "transparency and not part of the pass criterion."
        ),
        "eps": threshold,
        "details": details,
        "solo_baselines": solo,
    }


def run_temperature_invariance_probe(
    backend: Any,
    probes: list[dict[str, str]],
    *,
    t_low: float = 0.3,
    t_high: float = 1.0,
    eps: float | None = None,
) -> dict[str, Any]:
    """Wrap existing temperature-invariance probe for server/in-process backends."""
    same_t_values: list[float] = []
    max_cross = 0.0
    per_probe: list[dict[str, Any]] = []
    for probe in probes:
        diag = probe_temperature_invariance(
            backend,
            str(probe["prompt"]),
            t_low=t_low,
            t_high=t_high,
        )
        cross = diag.get("cross_t_dtle")
        same = diag.get("same_t_dtle")
        if isinstance(same, (int, float)):
            same_t_values.append(float(same))
        if isinstance(cross, (int, float)):
            max_cross = max(max_cross, float(cross))
        per_probe.append({"probe_id": probe.get("id"), **diag})
    threshold = float(eps) if eps is not None else resolve_tle_invariance_eps(same_t_values)
    return {
        "passed": max_cross <= threshold,
        "max_cross_t_dtle": max_cross,
        "eps": threshold,
        "per_probe": per_probe,
    }


def load_parity_probes(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("parity_prompts.json must be a list")
    return [x for x in raw if isinstance(x, dict) and x.get("prompt")]


def run_tle_invariance_probes(
    backend: Any,
    probes: list[dict[str, str]],
    *,
    max_concurrent_episodes: int,
    eps: float | None = None,
) -> dict[str, Any]:
    """TLE invariance under load: temperature + batch invariance at committed-action window."""
    temp = run_temperature_invariance_probe(backend, probes, eps=eps)
    batch = run_batch_invariance_probe(
        backend,
        probes,
        max_concurrent_episodes=max_concurrent_episodes,
        eps=temp.get("eps") if eps is None else eps,
    )
    return {
        "passed": bool(temp["passed"] and batch["passed"]),
        "temperature_invariance": temp,
        "batch_invariance": batch,
    }
