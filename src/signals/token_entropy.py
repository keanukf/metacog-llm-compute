"""
Token-Level Entropy (TLE) from model logprobs -- the study's first metacognitive signal (Factor 1),
the "processing-fluency" proxy H1 predicts is the *better*-calibrated of the two.

TLE is measured at the *committed-action* tokens only (never over C1/C2 reasoning), and this window
is identical across C0/C1/C2 -- a frozen, load-bearing invariant (docs/adrs.md): the action-window
slicing in ``extract_action_tle_from_response`` / ``slice_action_logprob_tokens`` is what enforces
it. Top-k is frozen at K=20 (Gate C-6 sensitivity sweep); ``mean_entropy_at_top_k`` supports the
variable-K re-analysis that sweep used.

- **Top-k distributions** (vLLM ``SamplingParams(logprobs=K)``, LM Studio
  ``/v1/responses``): Shannon entropy H = -sum_i p_i log2(p_i) over the
  **renormalized softmax** of the top-k logprobs only (approximation vs full vocabulary).
- **Top-1 only** (legacy / fallback): binary entropy from the chosen token's logprob.

**Scale invariant (cross-stage comparability):** TLE assumes temperature-invariant logprobs
as if T=1.0. The vLLM wrapper pins ``logprobs_mode="raw_logprobs"`` on the **engine**
(``LLM(...)``), not on ``SamplingParams`` — raw logprobs are returned before temperature /
top_k / top_p processing, so C2 diversity sampling temperature does not inflate entropy
relative to C0/C1.
"""

from __future__ import annotations

import math
from typing import Any


def _entropy_binary_from_top1_logprob(lp: float) -> float:
    """Legacy: treat single-token logprob as Bernoulli with p = exp(lp)."""
    p = math.exp(lp) if lp <= 0 else 0.0
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def entropy_shannon_from_top_logprobs(top: list[dict[str, Any]]) -> float:
    """
    Shannon entropy (bits) over renormalized softmax of top-k candidate logprobs.

    Each dict should have ``logprob`` (natural log of unnormalized mass or log p).
    We use log-sum-exp then normalize to a distribution on k atoms (approximation).
    """
    if not top:
        return 0.0
    lps: list[float] = []
    for x in top:
        if isinstance(x, dict) and x.get("logprob") is not None:
            lps.append(float(x["logprob"]))
    if not lps:
        return 0.0
    if len(lps) == 1:
        return 0.0
    m = max(lps)
    exps = [math.exp(lp - m) for lp in lps]
    s = sum(exps)
    if s <= 0:
        return 0.0
    probs = [e / s for e in exps]
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


def softmax_probs_from_top_logprobs(top: list[dict[str, Any]]) -> list[float]:
    """
    Renormalized probability masses over the top-k candidates (same softmax as Shannon TLE).
    Parallel order to ``top`` entries that have ``logprob``.
    """
    if not top:
        return []
    lps: list[float] = []
    for x in top:
        if isinstance(x, dict) and x.get("logprob") is not None:
            lps.append(float(x["logprob"]))
    if not lps:
        return []
    if len(lps) == 1:
        return [1.0]
    m = max(lps)
    exps = [math.exp(lp - m) for lp in lps]
    s = sum(exps)
    if s <= 0:
        n = len(lps)
        return [1.0 / n] * n
    return [e / s for e in exps]


def compute_tle(logprobs: list[dict[str, Any]] | list[float]) -> dict[str, float]:
    """
    Compute token-level entropy from a list of per-token records.

    If each dict has ``top_logprobs`` (list), uses Shannon entropy on that distribution
    per position. Otherwise uses legacy binary entropy from ``logprob`` only (vLLM/HF).

    Returns:
        Dict with ``mean_entropy`` and ``max_entropy`` over the sequence.
    """
    if not logprobs:
        return {"mean_entropy": 0.0, "max_entropy": 0.0}

    entropies: list[float] = []
    for x in logprobs:
        if isinstance(x, dict) and isinstance(x.get("top_logprobs"), list) and x["top_logprobs"]:
            entropies.append(entropy_shannon_from_top_logprobs(x["top_logprobs"]))
            continue
        if isinstance(x, dict):
            lp = x.get("logprob", 0.0)
            if isinstance(lp, (int, float)):
                entropies.append(_entropy_binary_from_top1_logprob(float(lp)))
            else:
                entropies.append(0.0)
            continue
        lp = float(x)
        entropies.append(_entropy_binary_from_top1_logprob(lp))

    if not entropies:
        return {"mean_entropy": 0.0, "max_entropy": 0.0}
    return {
        "mean_entropy": sum(entropies) / len(entropies),
        "max_entropy": max(entropies),
    }


def extract_tle_from_response(
    text: str,
    logprobs: list[dict[str, Any]] | list[float] | None,
) -> dict[str, float] | None:
    """
    Extract TLE from model response. If logprobs are missing, returns None.

    Args:
        text: Generated text (for length checks if needed).
        logprobs: Token logprobs from model_wrapper.generate(..., logprobs=True).

    Returns:
        compute_tle(logprobs) or None.
    """
    if logprobs is None:
        return None
    return compute_tle(logprobs)


def text_from_logprob_tokens(logprobs: list[dict[str, Any]]) -> str:
    """Reconstruct completion text from token records (for action-window slicing)."""
    return "".join(str(r.get("token", "")) for r in logprobs if isinstance(r, dict))


def slice_action_logprob_tokens(
    logprobs: list[dict[str, Any]] | list[float] | None,
    text: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return logprob records for the committed-action token window only.

    Matches ``extract_action_tle_from_response``: after ``</think>`` if present,
    otherwise the first non-empty line through its terminating newline (or end).
    """
    if logprobs is None or not logprobs:
        return []
    if not isinstance(logprobs, list):
        return []

    dict_rows = [x for x in logprobs if isinstance(x, dict)]
    if not dict_rows:
        return []

    has_token_text = all(isinstance(x.get("token"), str) for x in dict_rows)
    if not has_token_text:
        return dict_rows

    completion = text if text is not None else text_from_logprob_tokens(dict_rows)

    think_close_idx = (completion or "").lower().rfind("</think>")
    if think_close_idx >= 0:
        target_start = think_close_idx + len("</think>")
        while target_start < len(completion) and completion[target_start] in {
            "\n",
            "\r",
            " ",
            "\t",
        }:
            target_start += 1
        cursor = 0
        started = False
        action_slice: list[dict[str, Any]] = []
        for rec in dict_rows:
            if not isinstance(rec, dict):
                continue
            tok = str(rec.get("token", ""))
            start = cursor
            end = start + len(tok)
            cursor = end
            if end <= target_start:
                continue
            if not started:
                if tok.strip() == "":
                    continue
                started = True
            action_slice.append(rec)
            if started and ("\n" in tok):
                break
        return action_slice

    # No thinking block: the committed action is the first non-empty line, whether or
    # not the completion has further (post-action) lines after it.
    started = False
    action_slice = []
    for rec in dict_rows:
        if not isinstance(rec, dict):
            continue
        tok = str(rec.get("token", ""))
        if not started:
            if tok.strip() == "":
                continue
            started = True
        action_slice.append(rec)
        if started and ("\n" in tok):
            break
    return action_slice


def mean_entropy_at_top_k(
    logprob_tokens: list[dict[str, Any]],
    k: int,
) -> float | None:
    """Mean Shannon entropy over action tokens, each truncated to top-k logprobs."""
    if not logprob_tokens or k <= 0:
        return None
    entropies: list[float] = []
    for rec in logprob_tokens:
        if not isinstance(rec, dict):
            continue
        top = rec.get("top_logprobs")
        if not isinstance(top, list) or not top:
            continue
        trimmed = top[:k]
        if len(trimmed) < 2:
            continue
        entropies.append(entropy_shannon_from_top_logprobs(trimmed))
    if not entropies:
        return None
    return sum(entropies) / len(entropies)


def tle_mean_entropy_at_k_from_logprob_tokens(
    logprob_tokens: list[dict[str, Any]] | None,
    k: int,
    *,
    text: str | None = None,
) -> float | None:
    """Committed-action mean entropy at top-k (matches runtime TLE window, variable K)."""
    action_slice = slice_action_logprob_tokens(logprob_tokens, text=text)
    return mean_entropy_at_top_k(action_slice, k)


def extract_action_tle_from_response(
    text: str,
    logprobs: list[dict[str, Any]] | list[float] | None,
) -> dict[str, float] | None:
    """
    Extract TLE anchored to the executed action tokens only.

    If the completion contains a Qwen-style reasoning block (<think>...</think>),
    we treat the executed action as the first non-empty line **after** </think>.

    If token text is present in each record (``{"token": ...}``), we slice the logprob records
    up to and including the first newline *after* the first non-empty line begins.

    If token text is unavailable:
    - single-line completions: fall back to full-completion TLE (equivalent)
    - multi-line completions: return None (do not silently mix reasoning tokens into action TLE)
    """
    if logprobs is None:
        return None
    if not logprobs:
        return None

    has_token_text = isinstance(logprobs, list) and all(
        isinstance(x, dict) and isinstance(x.get("token"), str) for x in logprobs
    )
    out_text = (text or "").lstrip()

    if (text or "").lower().rfind("</think>") >= 0:
        if not has_token_text:
            return None
        action_slice = slice_action_logprob_tokens(logprobs, text=text)
        return compute_tle(action_slice) if action_slice else None

    is_multiline = "\n" in out_text
    if not has_token_text:
        if is_multiline:
            return None
        return compute_tle(logprobs)

    action_slice = slice_action_logprob_tokens(logprobs, text=text)
    if not action_slice:
        return None
    return compute_tle(action_slice)
