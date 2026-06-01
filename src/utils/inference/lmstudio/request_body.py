"""Build LM Studio ``POST /v1/responses`` request bodies with thinking controls."""

from __future__ import annotations

from typing import Any, Literal

# Wire format for POST /v1/responses (Open Responses spec; validated by LM Studio HTTP API).
OpenResponsesReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

# LM Studio runtime (model layer) uses on/off — see dev log when sending unsupported values
# like "medium" on qwen/qwen3-4b. Map via Open Responses effort on the wire only.


def reasoning_effort_for_enable_thinking(
    enable_thinking: bool,
    *,
    thinking_effort_when_on: OpenResponsesReasoningEffort = "low",
) -> OpenResponsesReasoningEffort:
    """
    Map repo ``enable_thinking`` to ``reasoning.effort`` on the HTTP request.

    - ``False`` → ``"none"`` (LM Studio maps to model reasoning **off**; ``reasoning_tokens`` ≈ 0).
    - ``True`` → ``"low"`` by default (maps to model reasoning **on**). Avoid ``"medium"`` on
      Qwen3-4B: LM Studio logs that only **on**/**off** are supported and coerces ``medium`` → ``on``.

    The API rejects bare ``"on"``/``"off"`` (invalid enum); use Open Responses values only.
    """
    if enable_thinking:
        return thinking_effort_when_on
    return "none"


def thinking_control_fields(
    enable_thinking: bool | None,
    *,
    thinking_effort_when_on: OpenResponsesReasoningEffort = "low",
) -> dict[str, Any]:
    """
    Fields to merge into a ``/v1/responses`` JSON body for Qwen3 / LM Studio.

    Three layers when ``enable_thinking`` is set:

    - ``reasoning.effort`` — Open Responses enum on the wire (``none`` / ``low`` / …).
    - ``enable_thinking`` + ``chat_template_kwargs`` — Qwen Jinja ``enable_thinking``.
    """
    if enable_thinking is None:
        return {}
    et = bool(enable_thinking)
    effort = reasoning_effort_for_enable_thinking(
        et, thinking_effort_when_on=thinking_effort_when_on
    )
    return {
        "enable_thinking": et,
        "chat_template_kwargs": {"enable_thinking": et},
        "reasoning": {"effort": effort},
    }


def build_v1_responses_body(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_logprobs: int,
    enable_thinking: bool | None = None,
    thinking_effort_when_on: OpenResponsesReasoningEffort = "low",
    include_logprobs: bool = True,
    extra_input_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble a complete ``POST /v1/responses`` payload."""
    input_messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    if extra_input_messages:
        input_messages = [*extra_input_messages, *input_messages]

    body: dict[str, Any] = {
        "model": model,
        "input": input_messages,
        "temperature": float(temperature),
        "max_output_tokens": int(max_tokens),
    }
    if include_logprobs:
        body["include"] = ["message.output_text.logprobs"]
        body["top_logprobs"] = int(top_logprobs)
    body.update(
        thinking_control_fields(
            enable_thinking,
            thinking_effort_when_on=thinking_effort_when_on,
        )
    )
    return body


def summarize_responses_usage(data: dict[str, Any]) -> dict[str, Any]:
    """Extract reasoning vs message token counts and output block types from a response dict."""
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    out_details = usage.get("output_tokens_details") if isinstance(usage, dict) else {}
    reasoning_tokens = 0
    if isinstance(out_details, dict) and out_details.get("reasoning_tokens") is not None:
        reasoning_tokens = int(out_details["reasoning_tokens"])

    output_block_types: list[str] = []
    has_message = False
    message_preview = ""
    if isinstance(data.get("output"), list):
        for block in data["output"]:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type") or "unknown")
            output_block_types.append(btype)
            if btype == "message":
                has_message = True
                for part in block.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        message_preview = str(part.get("text") or "").strip()[:120]

    resp_reasoning = data.get("reasoning")
    effort = None
    if isinstance(resp_reasoning, dict):
        effort = resp_reasoning.get("effort")

    return {
        "output_block_types": output_block_types,
        "has_message_block": has_message,
        "message_preview": message_preview,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": int(usage.get("output_tokens") or 0) if isinstance(usage, dict) else 0,
        "response_reasoning_effort": effort,
    }
