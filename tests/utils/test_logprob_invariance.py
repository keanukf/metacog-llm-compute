"""Token-level-entropy temperature-invariance diagnostics (thesis §5.7).

Verifies the first-token TLE matches the entropy helper, the probe passes for identical
distributions, and the invariance epsilon resolves to the preregistered floor and scales with
same-temperature noise. TLE must reflect the model's uncertainty, not the sampling temperature, or
it would be a temperature artefact rather than the metacognitive signal RQ1 studies; this probe is
the guard that keeps that assumption honest.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.signals.token_entropy import entropy_shannon_from_top_logprobs
from src.utils.inference.logprob_invariance import (
    TLE_INVARIANCE_EPS_BITS,
    first_token_tle,
    max_top_logprob_delta,
    probe_temperature_invariance,
    resolve_tle_invariance_eps,
)


def _top(k: int = 5, base: float = -1.0) -> list[dict]:
    return [{"token": f"t{i}", "logprob": base - 0.1 * i} for i in range(k)]


class _InvariantModel:
    def __init__(self, top: list[dict]) -> None:
        self._top = top

    def generate(self, prompt: str, **kwargs):
        _ = prompt, kwargs
        return "tok", [{"token": "tok", "logprob": -0.2, "top_logprobs": list(self._top)}]


def test_first_token_tle_matches_entropy_helper() -> None:
    top = _top()
    lp = [{"token": "tok", "top_logprobs": top}]
    assert first_token_tle(lp) == entropy_shannon_from_top_logprobs(top)


def test_probe_temperature_invariance_passes_for_identical_distributions() -> None:
    model = _InvariantModel(_top())
    diag = probe_temperature_invariance(model, "p", t_low=0.3, t_high=1.0)
    assert diag["cross_t_dtle"] == 0.0
    assert diag["same_t_dtle"] == 0.0
    assert diag["max_logprob_delta_cross_t"] == 0.0


def test_resolve_tle_invariance_eps_uses_preregistered_floor() -> None:
    assert resolve_tle_invariance_eps([0.001]) == TLE_INVARIANCE_EPS_BITS


def test_resolve_tle_invariance_eps_scales_with_same_t_noise() -> None:
    eps = resolve_tle_invariance_eps([0.02, 0.01])
    assert eps >= TLE_INVARIANCE_EPS_BITS
    assert eps >= 0.02 * 3.0


def test_max_top_logprob_delta_on_shared_keys() -> None:
    a = [{"token": "a", "logprob": -1.0}, {"token": "b", "logprob": -2.0}]
    b = [{"token": "a", "logprob": -1.05}, {"token": "b", "logprob": -2.0}]
    assert max_top_logprob_delta(a, b) == pytest.approx(0.05)


def test_probe_reports_scaling_diagnostic() -> None:
    model = MagicMock()
    top = _top()
    model.generate.return_value = ("x", [{"token": "x", "top_logprobs": top}])
    diag = probe_temperature_invariance(model, "p")
    assert diag["predicted_scaling_logprob_span"] is not None
    assert diag["predicted_scaling_logprob_span"] > 0.0
