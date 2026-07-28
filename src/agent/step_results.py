"""Backward-compatible unpacking of step-function return tuples.

Step functions have grown their return arity over the project's life (from a 3-tuple to the
current 11-tuple ``StepReturn``). This adapter lets the episode loop accept any historical width
-- older mocks and tests still return short tuples -- and always hand back the full 11-field shape
with missing trailing fields filled in. New code should return the full ``StepReturn``.
"""

from __future__ import annotations

from typing import Any


def normalize_step_result(
    result: tuple,
) -> tuple[
    str,
    dict | None,
    float | None,
    int,
    int,
    list[dict[str, Any]] | None,
    dict[str, Any] | None,
    str | None,
    str | None,
    dict[str, Any] | None,
    int,
]:
    """
    Unpack step result as (action, tle, vc, tokens_used, lm_calls_this_step, logprobs_raw,
    vc_detail, prompt_full, response_full, call_detail, prompt_tokens_used).
    """
    raw_lp: list[dict[str, Any]] | None = None
    vc_detail: dict[str, Any] | None = None
    prompt_full: str | None = None
    response_full: str | None = None
    call_detail: dict[str, Any] | None = None
    n = len(result)
    prompt_tokens_used = 0
    if n >= 11:
        p10 = result[10]
        prompt_tokens_used = int(p10) if isinstance(p10, (int, float)) else 0
    if n >= 9:
        p7, p8 = result[7], result[8]
        prompt_full = p7 if isinstance(p7, str) else None
        response_full = p8 if isinstance(p8, str) else None
    if n >= 10:
        p9 = result[9]
        call_detail = p9 if isinstance(p9, dict) else None
    if n >= 7:
        fifth, sixth = result[5], result[6]
        raw_lp = fifth if isinstance(fifth, list) else None
        vc_detail = sixth if isinstance(sixth, dict) else None
    elif n == 6:
        sixth = result[5]
        if isinstance(sixth, dict) and (
            "vc_prompt" in sixth or "vc_raw_text" in sixth or "vc_value" in sixth
        ):
            vc_detail = sixth
        elif isinstance(sixth, list):
            raw_lp = sixth
    if n >= 5:
        return (
            result[0],
            result[1],
            result[2],
            int(result[3]),
            int(result[4]),
            raw_lp,
            vc_detail,
            prompt_full,
            response_full,
            call_detail,
            prompt_tokens_used,
        )
    if len(result) >= 4:
        return (
            result[0],
            result[1],
            result[2],
            int(result[3]),
            1,
            raw_lp,
            vc_detail,
            prompt_full,
            response_full,
            call_detail,
            prompt_tokens_used,
        )
    return (
        result[0],
        result[1],
        result[2],
        0,
        1,
        raw_lp,
        vc_detail,
        prompt_full,
        response_full,
        call_detail,
        prompt_tokens_used,
    )
