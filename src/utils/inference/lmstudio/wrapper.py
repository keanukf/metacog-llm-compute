"""LM Studio wrapper: responses-only path with per-call diagnostics."""

from __future__ import annotations

from typing import Any

from src.utils.errors import BackendError
from src.utils.inference.base import ModelWrapper
from src.utils.inference.lmstudio.parse import (
    build_lmstudio_call_diagnostics,
    extract_reasoning_and_message,
    parse_lmstudio_responses_json,
)
from src.utils.inference.lmstudio.request_body import reasoning_effort_for_enable_thinking
from src.utils.inference.lmstudio.responses import (
    default_api_key,
    post_v1_responses,
    resolve_lmstudio_api_host,
)
from src.utils.inference.urls import normalize_openai_base_url, responses_endpoint_url


def attach_lmstudio_diagnostics_to_subcalls(
    model: Any,
    subcalls: list[dict[str, Any]],
) -> None:
    """Attach recent LM Studio call diagnostics to subcall dicts (by order)."""
    consume = getattr(model, "consume_call_diagnostics", None)
    if not callable(consume):
        return
    diags = consume()
    for idx, sc in enumerate(subcalls):
        if idx < len(diags):
            sc["lmstudio"] = diags[idx]


def collect_step_inference_diagnostics(model: Any) -> list[dict[str, Any]] | None:
    """Return diagnostics for the current step (all LM calls since last consume)."""
    consume = getattr(model, "consume_call_diagnostics", None)
    if not callable(consume):
        return None
    diags = consume()
    return diags if diags else None


class LMStudioWrapper(ModelWrapper):
    """
    LM Studio local server via official SDK host discovery + ``POST /v1/responses`` only.

    No ``/v1/completions`` fallback. Token logprobs require ``include: message.output_text.logprobs``.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str | None = None,
        *,
        top_logprobs: int = 20,
        lmstudio_top_logprobs: int | None = None,
        api_host: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._base_url = normalize_openai_base_url(base_url.strip())
        self._api_key = api_key
        if lmstudio_top_logprobs is not None:
            k = int(lmstudio_top_logprobs)
        else:
            k = int(top_logprobs)
        self._top_logprobs = max(1, k)
        self._api_host = api_host
        self._kwargs = kwargs
        self._resolved_api_host: str | None = None
        self._pending_diagnostics: list[dict[str, Any]] = []

    @property
    def responses_endpoint(self) -> str:
        return responses_endpoint_url(self._base_url)

    def _ensure_api_host(self) -> str:
        if self._resolved_api_host is None:
            self._resolved_api_host = resolve_lmstudio_api_host(
                self._base_url,
                explicit_api_host=self._api_host,
            )
        return self._resolved_api_host

    def consume_call_diagnostics(self) -> list[dict[str, Any]]:
        """Return and clear diagnostics recorded since the previous consume."""
        out = list(self._pending_diagnostics)
        self._pending_diagnostics.clear()
        return out

    def _record_call(
        self,
        *,
        data: dict[str, Any] | None,
        reasoning_text: str,
        message_text: str,
        token_records: list[dict[str, Any]],
        assembled_text: str,
        logprobs_requested: bool,
        enable_thinking: bool | None,
        http_status: int | None,
        error: str | None,
    ) -> None:
        effort: str | None = None
        if isinstance(enable_thinking, bool):
            effort = reasoning_effort_for_enable_thinking(enable_thinking)
        self._pending_diagnostics.append(
            build_lmstudio_call_diagnostics(
                endpoint=self.responses_endpoint,
                data=data,
                reasoning_text=reasoning_text,
                message_text=message_text,
                token_records=token_records,
                assembled_text=assembled_text,
                logprobs_requested=logprobs_requested,
                enable_thinking=enable_thinking,
                reasoning_effort=effort,
                http_status=http_status,
                error=error,
            )
        )

    def generate(
        self,
        prompt: str,
        *,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        self._ensure_api_host()
        enable_thinking = kwargs.pop("enable_thinking", None)
        key = default_api_key(self._api_key)

        data, http_status, err = post_v1_responses(
            base_url=self._base_url,
            api_key=key,
            model=self._model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_logprobs=self._top_logprobs,
            enable_thinking=enable_thinking if isinstance(enable_thinking, bool) else None,
            include_logprobs=bool(logprobs),
        )

        if data is None:
            self._record_call(
                data=None,
                reasoning_text="",
                message_text="",
                token_records=[],
                assembled_text="",
                logprobs_requested=bool(logprobs),
                enable_thinking=enable_thinking if isinstance(enable_thinking, bool) else None,
                http_status=http_status,
                error=err or "responses request failed",
            )
            if err:
                raise BackendError(f"LM Studio POST /v1/responses failed: {err}")
            return "", None

        reasoning_text, message_text, token_records, _ = extract_reasoning_and_message(data)
        text, lp_list = parse_lmstudio_responses_json(data, enable_thinking=enable_thinking)
        self._record_call(
            data=data,
            reasoning_text=reasoning_text,
            message_text=message_text,
            token_records=token_records,
            assembled_text=text,
            logprobs_requested=bool(logprobs),
            enable_thinking=enable_thinking if isinstance(enable_thinking, bool) else None,
            http_status=http_status,
            error=None,
        )
        return text, (lp_list if logprobs else None)
