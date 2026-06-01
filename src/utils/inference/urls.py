"""OpenAI-compatible base URL helpers."""

from __future__ import annotations

from urllib.parse import urlparse


def normalize_openai_base_url(base_url: str) -> str:
    """
    OpenAI-compatible clients expect base_url ending with /v1.

    Accepts either a server root (e.g. http://host:1234) or full API base (…/v1).
    """
    bu = base_url.strip().rstrip("/")
    if bu.endswith("/v1"):
        return bu
    return f"{bu}/v1"


def openai_base_url_to_api_host(base_url: str) -> str:
    """Convert ``http://host:port/v1`` to LM Studio SDK ``api_host`` (``host:port``)."""
    normalized = normalize_openai_base_url(base_url)
    parsed = urlparse(normalized)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is not None:
        return f"{host}:{parsed.port}"
    if parsed.scheme == "https":
        return f"{host}:443"
    return f"{host}:80"


def responses_endpoint_url(base_url: str) -> str:
    """Full URL for ``POST /v1/responses``."""
    return f"{normalize_openai_base_url(base_url).rstrip('/')}/responses"
