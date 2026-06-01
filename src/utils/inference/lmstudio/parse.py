"""Parse LM Studio ``POST /v1/responses`` JSON and build call diagnostics."""

from __future__ import annotations

from typing import Any


def _consume_logprobs_list(raw: Any, token_records: list[dict[str, Any]]) -> None:
    if not isinstance(raw, list):
        return
    for tok in raw:
        if not isinstance(tok, dict):
            continue
        rec: dict[str, Any] = {
            "token": str(tok.get("token", "")),
            "logprob": float(tok.get("logprob", 0.0)),
        }
        top = tok.get("top_logprobs")
        if isinstance(top, list) and top:
            rec["top_logprobs"] = []
            for x in top:
                if isinstance(x, dict):
                    rec["top_logprobs"].append(
                        {
                            "token": str(x.get("token", "")),
                            "logprob": float(x.get("logprob", 0.0)),
                        }
                    )
        token_records.append(rec)


def extract_reasoning_and_message(
    data: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    """
    Split LM Studio /v1/responses ``output`` into reasoning, message, logprobs, block types.
    """
    reasoning_chunks: list[str] = []
    message_chunks: list[str] = []
    token_records: list[dict[str, Any]] = []
    output_block_types: list[str] = []

    def _walk_parts(parts: Any) -> None:
        if not isinstance(parts, list):
            return
        for part in parts:
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type") or "")
            t = part.get("text")
            if ptype in ("reasoning_text", "reasoning") and isinstance(t, str) and t.strip():
                reasoning_chunks.append(t.strip())
            elif ptype == "output_text" or (ptype == "" and "text" in part):
                if isinstance(t, str) and t:
                    message_chunks.append(t)
                _consume_logprobs_list(part.get("logprobs"), token_records)

    out_blocks = data.get("output")
    if isinstance(out_blocks, list):
        for block in out_blocks:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type") or "unknown")
            output_block_types.append(btype)
            if btype == "reasoning":
                _walk_parts(block.get("content"))
                continue
            if btype == "message":
                _walk_parts(block.get("content"))
                continue
            if btype == "output_text":
                t = block.get("text")
                if isinstance(t, str) and t:
                    message_chunks.append(t)
                _consume_logprobs_list(block.get("logprobs"), token_records)
                continue
            _walk_parts(block.get("content"))

    reasoning_text = "\n\n".join(reasoning_chunks).strip()
    message_text = "".join(message_chunks).strip()
    return reasoning_text, message_text, token_records, output_block_types


def _assemble_assistant_text(
    reasoning_text: str,
    message_text: str,
    *,
    enable_thinking: bool | None,
) -> str:
    if enable_thinking is False:
        return message_text
    if reasoning_text and message_text:
        return f"<think>\n{reasoning_text}\n</think>\n\n{message_text.lstrip()}"
    if reasoning_text:
        return f"<think>\n{reasoning_text}\n</think>\n\n"
    return message_text


def parse_lmstudio_responses_json(
    data: dict[str, Any],
    *,
    enable_thinking: bool | None = None,
) -> tuple[str, list[dict[str, Any]] | None]:
    """
    Parse LM Studio POST /v1/responses JSON into (assistant_text, per_token_records).
    """
    if not isinstance(data, dict):
        return "", None
    reasoning_text, message_text, token_records, _ = extract_reasoning_and_message(data)
    use_thinking = enable_thinking
    if use_thinking is None and reasoning_text:
        use_thinking = True
    text = _assemble_assistant_text(
        reasoning_text,
        message_text,
        enable_thinking=use_thinking,
    )
    if not text and token_records:
        text = "".join(rec.get("token", "") for rec in token_records).strip()
    if not token_records:
        return text, None
    return text, token_records


def build_lmstudio_call_diagnostics(
    *,
    endpoint: str,
    data: dict[str, Any] | None,
    reasoning_text: str,
    message_text: str,
    token_records: list[dict[str, Any]],
    assembled_text: str,
    logprobs_requested: bool,
    enable_thinking: bool | None,
    reasoning_effort: str | None = None,
    http_status: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Structured per-call metadata for traces and debug views."""
    output_block_types: list[str] = []
    if isinstance(data, dict):
        _, _, _, output_block_types = extract_reasoning_and_message(data)

    status = "ok"
    if error:
        status = "http_error" if http_status else "request_failed"
    elif not assembled_text and reasoning_text and not message_text:
        status = "reasoning_only"
    elif not assembled_text:
        status = "empty_message"
    elif logprobs_requested and not token_records:
        status = "missing_logprobs"

    return {
        "route": "POST /v1/responses",
        "endpoint": endpoint,
        "method": "POST",
        "output_block_types": output_block_types,
        "has_message_text": bool(message_text),
        "has_reasoning_text": bool(reasoning_text),
        "has_token_logprobs": bool(token_records),
        "logprobs_requested": bool(logprobs_requested),
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "status": status,
        "http_status": http_status,
        "error": error,
    }
