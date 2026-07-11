"""Tests for batch-invariance probe logic (mock backend, no GPU)."""

from __future__ import annotations

from src.execution.parity import run_batch_invariance_probe
from src.signals.token_entropy import extract_action_tle_from_response


class _StableBackend:
    """Returns identical logprobs regardless of load."""

    def generate(self, prompt, logprobs=False, **kwargs):
        text = "go north"
        lp = [
            {
                "token": "go",
                "logprob": 0.0,
                "top_logprobs": [
                    {"token": "go", "logprob": 0.0},
                    {"token": "look", "logprob": -2.0},
                ],
            },
            {
                "token": " north",
                "logprob": 0.0,
                "top_logprobs": [
                    {"token": " north", "logprob": 0.0},
                    {"token": " south", "logprob": -3.0},
                ],
            },
        ]
        return text, lp if logprobs else None


def test_action_window_tle_on_mock_records():
    text, lp = _StableBackend().generate("p", logprobs=True)
    tle = extract_action_tle_from_response(text, lp)
    assert tle is not None
    assert "mean_entropy" in tle


def test_batch_invariance_passes_when_stable():
    probes = [{"id": "p1", "prompt": "go north"}]
    report = run_batch_invariance_probe(
        _StableBackend(),
        probes,
        max_concurrent_episodes=2,
        eps=0.05,
        max_tokens=8,
    )
    assert report["passed"] is True
    assert report["max_dtle"] <= 0.05


class _DriftBackend(_StableBackend):
    _calls = 0

    def generate(self, prompt, logprobs=False, **kwargs):
        _DriftBackend._calls += 1
        if _DriftBackend._calls > 2:
            text = "go north"
            lp = [
                {
                    "token": "go",
                    "logprob": 0.0,
                    "top_logprobs": [
                        {"token": "go", "logprob": 0.0},
                        {"token": "look", "logprob": -0.01},
                    ],
                }
            ]
            return text, lp if logprobs else None
        return super().generate(prompt, logprobs=logprobs, **kwargs)


def test_batch_invariance_fails_when_dtle_exceeds_eps():
    _DriftBackend._calls = 0
    probes = [{"id": "p1", "prompt": "go north"}]
    report = run_batch_invariance_probe(
        _DriftBackend(),
        probes,
        max_concurrent_episodes=2,
        eps=0.001,
        max_tokens=8,
    )
    assert report["passed"] is False
    assert report["max_dtle"] > 0.001
