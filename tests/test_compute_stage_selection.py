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

