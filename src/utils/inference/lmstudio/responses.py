"""LM Studio /v1/responses HTTP client (httpx, SDK host discovery)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from src.utils.errors import BackendError
from src.utils.inference.lmstudio.request_body import build_v1_responses_body
from src.utils.inference.urls import openai_base_url_to_api_host, responses_endpoint_url


def resolve_lmstudio_api_host(
    base_url: str,
    *,
    explicit_api_host: str | None = None,
) -> str:
    """
    Resolve LM Studio ``api_host`` (``host:port``) using the official SDK probe when possible.
    """
    if explicit_api_host:
        host = explicit_api_host.strip()
        try:
            from lmstudio import Client

            if Client.is_valid_api_host(host):
                return host
        except ImportError as e:
            raise BackendError(
                "LMStudioWrapper requires the lmstudio package. "
                "Install with: pip install 'metacog-llm-compute[full]' or pip install lmstudio"
            ) from e
        raise BackendError(f"LM Studio API host is not reachable: {host!r}")

    derived = openai_base_url_to_api_host(base_url)
    try:
        from lmstudio import Client

        if Client.is_valid_api_host(derived):
            return derived
        found = Client.find_default_local_api_host()
        if found:
            return str(found)
    except ImportError as e:
        raise BackendError(
            "LMStudioWrapper requires the lmstudio package. Install with: pip install lmstudio"
        ) from e

    raise BackendError(
        f"LM Studio server not reachable at {derived!r} (from base_url={base_url!r}). "
        "Start LM Studio local server or set inference.lmstudio_base_url / LM_STUDIO_BASE_URL."
    )


def post_v1_responses(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_logprobs: int,
    enable_thinking: bool | None = None,
    include_logprobs: bool = True,
    timeout_s: float = 600.0,
) -> tuple[dict[str, Any] | None, int | None, str | None]:
    """
    POST JSON to ``{base}/responses``.

    Returns ``(parsed_dict, http_status, error_message)``.
    """
    url = responses_endpoint_url(base_url)
    body = build_v1_responses_body(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_logprobs=top_logprobs,
        enable_thinking=enable_thinking,
        include_logprobs=include_logprobs,
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    def _post(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, int | None, str | None]:
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            return None, None, str(exc)
        if resp.status_code >= 400:
            err_body = (resp.text or "")[:800]
            return None, resp.status_code, err_body or f"HTTP {resp.status_code}"
        try:
            out = resp.json()
        except ValueError as exc:
            return None, resp.status_code, f"invalid JSON: {exc}"
        return (out if isinstance(out, dict) else None), resp.status_code, None

    data, status, err = _post(body)
    if data is not None or err is None:
        return data, status, err

    # Retry once with max_tokens if server rejects max_output_tokens
    if status == 400 and "max_output_tokens" in body:
        body2 = dict(body)
        body2.pop("max_output_tokens", None)
        body2["max_tokens"] = int(max_tokens)
        return _post(body2)
    return data, status, err


def default_api_key(explicit: str | None) -> str:
    return explicit or os.environ.get("LM_STUDIO_API_KEY") or "lm-studio"
