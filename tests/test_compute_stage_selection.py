"""Tests for resolving which compute stages a Phase-1 domain runs.

Ensures the config-to-stages resolution accepts the several YAML spellings (int count, string, list),
lets a per-domain override beat the global default, and rejects malformed values -- including the
YAML bareword-bool trap (``on``/``off`` parsing to True/False) that could silently corrupt the C0/
C1/C2 worklist.
"""

from __future__ import annotations

import pytest

from src.utils.compute_stage_selection import resolve_compute_stages_for_domain


def test_resolve_compute_stages_int_count() -> None:
    cfg = {"pilot": {"compute_stages": 1}}
    assert resolve_compute_stages_for_domain(cfg, domain="textworld") == ["C0"]
    cfg = {"pilot": {"compute_stages": 2}}
    assert resolve_compute_stages_for_domain(cfg, domain="textworld") == ["C0", "C1"]
    cfg = {"pilot": {"compute_stages": 3}}
    assert resolve_compute_stages_for_domain(cfg, domain="textworld") == ["C0", "C1", "C2"]


def test_resolve_compute_stages_string_and_list() -> None:
    cfg = {"pilot": {"compute_stages": "C0"}}
    assert resolve_compute_stages_for_domain(cfg, domain="textworld") == ["C0"]
    cfg = {"pilot": {"compute_stages": "C0, C2"}}
    assert resolve_compute_stages_for_domain(cfg, domain="textworld") == ["C0", "C2"]
    cfg = {"pilot": {"compute_stages": ["C2", "C0"]}}
    # Ordered canonically.
    assert resolve_compute_stages_for_domain(cfg, domain="textworld") == ["C0", "C2"]


def test_resolve_compute_stages_by_domain_overrides_global() -> None:
    cfg = {
        "pilot": {
            "compute_stages": 3,
            "compute_stages_by_domain": {"textworld": "C0", "tower_of_hanoi": ["C1"]},
        }
    }
    assert resolve_compute_stages_for_domain(cfg, domain="textworld") == ["C0"]
    assert resolve_compute_stages_for_domain(cfg, domain="tower_of_hanoi") == ["C1"]


def test_resolve_compute_stages_unknown_raises() -> None:
    cfg = {"pilot": {"compute_stages": ["C9"]}}
    with pytest.raises(ValueError):
        resolve_compute_stages_for_domain(cfg, domain="textworld")


def test_resolve_compute_stages_rejects_yaml_bareword_bool() -> None:
    """
    `compute_stages` is documented as int|str|list, never bool — but bool is a subclass of
    int in Python, and YAML 1.1 parses a bare `no`/`off` as False and `yes`/`on` as True. Without
    an explicit guard, `compute_stages: no` would silently fall through to `n=0` -> `[]` -> the
    caller's default stage list (all 3 stages run instead of erroring), and
    `compute_stages: yes`/`true` would silently resolve to just `["C0"]` (n=1). Both are the
    kind of silent misconfiguration this repo's YAML bareword traps keep producing elsewhere
    (see logprob_sidecar_mode: off, docs/gate_e_rehearsal.md) — reject instead of guessing.
    """
    cfg_false = {"pilot": {"compute_stages": False}}
    with pytest.raises(TypeError):
        resolve_compute_stages_for_domain(cfg_false, domain="textworld")

    cfg_true = {"pilot": {"compute_stages": True}}
    with pytest.raises(TypeError):
        resolve_compute_stages_for_domain(cfg_true, domain="textworld")

    cfg_by_domain = {
        "pilot": {
            "compute_stages": 3,
            "compute_stages_by_domain": {"textworld": False},
        }
    }
    with pytest.raises(TypeError):
        resolve_compute_stages_for_domain(cfg_by_domain, domain="textworld")
