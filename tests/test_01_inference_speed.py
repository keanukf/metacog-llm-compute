"""
Pilot Test 1 — Inferenzgeschwindigkeit.
With a mock model, run 50 "prompts", compute tok/s and latency structure.
"""
from __future__ import annotations

import time

import pytest


def _run_mock_benchmark(num_prompts: int, tokens_per_call: int = 200):
    """Simulate 50 prompts with fixed tokens and timings; return result dict."""
    latencies = []
    total_tokens = 0
    for _ in range(num_prompts):
        t0 = time.perf_counter()
        # Simulate work
        time.sleep(0.001)
        total_tokens += tokens_per_call
        latencies.append(time.perf_counter() - t0)
    elapsed = sum(latencies)
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    return {
        "tokens_per_sec": tokens_per_sec,
        "latency_mean": sum(latencies) / len(latencies) if latencies else 0,
        "latency_std": (sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)) ** 0.5 if latencies else 0,
        "total_tokens": total_tokens,
        "num_prompts": num_prompts,
    }


def test_inference_speed_result_structure():
    """Result dict has required keys: tokens_per_sec, latency_mean."""
    result = _run_mock_benchmark(50)
    assert "tokens_per_sec" in result
    assert "latency_mean" in result
    assert result["num_prompts"] == 50
    assert result["total_tokens"] == 50 * 200


def test_inference_speed_tokens_per_sec_positive():
    """With mock timing, tokens_per_sec is positive."""
    result = _run_mock_benchmark(10, tokens_per_call=100)
    assert result["tokens_per_sec"] > 0
