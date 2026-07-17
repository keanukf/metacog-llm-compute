"""
Compute-stage selection utilities.

The codebase historically used numeric stage counts (1/2/3) to mean:
  1 -> C0
  2 -> C0,C1
  3 -> C0,C1,C2

For pilots, we also want explicit selection (e.g. "C0" only) and per-domain overrides.
This module resolves those configuration variants into a canonical ordered list of stage
labels drawn from {"C0","C1","C2"}.
"""

from __future__ import annotations

from typing import Any, Iterable

_VALID_STAGES: tuple[str, ...] = ("C0", "C1", "C2")


def _dedupe_preserve_order(xs: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _normalize_stage_token(tok: str) -> str:
    t = tok.strip().upper()
    if not t:
        return ""
    if t in {"0", "C0"}:
        return "C0"
    if t in {"1", "C1"}:
        return "C1"
    if t in {"2", "C2"}:
        return "C2"
    return t


def _parse_stage_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        # bool is a subclass of int in Python, and YAML 1.1 parses bare `yes`/`no`/`on`/`off`
        # as booleans — without this guard, `compute_stages: no` would silently fall through
        # the int branch below (n=0 -> [] -> caller falls back to its default list) and
        # `compute_stages: yes`/`true` would silently resolve to just ["C0"] (n=1), neither of
        # which is a documented value for this field. Reject explicitly instead of guessing.
        raise TypeError(
            f"compute_stages must be int|str|list|tuple, got bool {value!r} "
            "(YAML bareword yes/no/on/off parses as a Python bool — use an int (1..3) or an "
            'explicit stage string/list instead, e.g. "C0" or ["C0", "C2"])'
        )
    if isinstance(value, int):
        n = int(value)
        if n <= 0:
            return []
        if n > 3:
            raise ValueError(f"compute_stages int must be 1..3, got {n}")
        return list(_VALID_STAGES[:n])
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # Back-compat: allow "3" / "2" / "1" as strings.
        if s.isdigit():
            return _parse_stage_list(int(s))
        parts = [p for p in s.replace(",", " ").split() if p.strip()]
        toks = [_normalize_stage_token(p) for p in parts]
        return [t for t in toks if t]
    if isinstance(value, (list, tuple)):
        toks = []
        for x in value:
            if x is None:
                continue
            toks.append(_normalize_stage_token(str(x)))
        return [t for t in toks if t]
    raise TypeError(f"compute_stages must be int|str|list|tuple|dict, got {type(value).__name__}")


def _validate_and_order(stages: list[str]) -> list[str]:
    if not stages:
        return []
    stages = _dedupe_preserve_order(stages)
    unknown = [s for s in stages if s not in _VALID_STAGES]
    if unknown:
        raise ValueError(f"Unknown compute stage(s): {unknown}. Valid: {list(_VALID_STAGES)}")
    # Keep canonical order C0,C1,C2 regardless of input ordering.
    return [s for s in _VALID_STAGES if s in set(stages)]


def resolve_compute_stages_for_domain(
    config: dict[str, Any],
    *,
    domain: str | None,
    default: Iterable[str] = _VALID_STAGES,
    config_key: str = "pilot",
) -> list[str]:
    """
    Resolve enabled compute stages for a given domain.

    Supported config shapes (under config[config_key], typically ``pilot`` or ``phase1``):
      - compute_stages: 3 | 2 | 1 (or "3"/"2"/"1")
      - compute_stages: "C0" | "C0,C2" | ["C0","C2"]
      - compute_stages_by_domain: {textworld: "C0", tower_of_hanoi: ["C0","C1"]}
      - compute_stages: {textworld: "C0"}  (legacy-ish convenience)
    """
    section = config.get(config_key) or {}
    default_list = _validate_and_order(list(default))

    by_dom = section.get("compute_stages_by_domain")
    if domain and isinstance(by_dom, dict) and domain in by_dom:
        stages = _validate_and_order(_parse_stage_list(by_dom[domain]))
        return stages if stages else default_list

    raw = section.get("compute_stages")
    if isinstance(raw, dict) and domain and domain in raw:
        stages = _validate_and_order(_parse_stage_list(raw[domain]))
        return stages if stages else default_list

    stages = _validate_and_order(_parse_stage_list(raw))
    return stages if stages else default_list
