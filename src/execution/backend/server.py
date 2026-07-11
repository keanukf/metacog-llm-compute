"""HTTP client backend for vLLM OpenAI-compatible server."""

from __future__ import annotations

import re
import threading
from typing import Any

import httpx

from src.utils.errors import BackendError
from src.utils.inference.logprob_config import resolve_top_logprobs
from src.utils.inference.vllm_shared import normalize_chat_completion_logprobs

_THINKING_MARKERS = (
    re.compile(r"<think>", re.IGNORECASE),
    re.compile(r"</think>", re.IGNORECASE),
)


class ServerBackend:
    """
    Sync HTTP client for ``vllm serve`` ``/v1/chat/completions``.

    Server applies chat template; client sends ``messages`` only.
    """

    def __init__(
        self,
        *,
        server_url: str,
        model_name: str,
        top_logprobs: int = 20,
        timeout_s: float = 300.0,
    ) -> None:
        base = server_url.rstrip("/")
        if base.endswith("/v1"):
            self._url = f"{base}/chat/completions"
        else:
            self._url = f"{base}/v1/chat/completions"
        self._model_name = model_name
        self._top_logprobs = max(1, int(top_logprobs))
        self._timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=self._timeout_s)

    @property
    def model_name(self) -> str:
        return self._model_name

    def close(self) -> None:
        with self._lock:
            self._client.close()

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            resp = self._client.post(self._url, json=payload)
        if resp.status_code >= 400:
            raise BackendError(f"vLLM server HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        if not isinstance(data, dict):
            raise BackendError("vLLM server returned non-object JSON")
        return data

    def _build_payload(
        self,
        prompt: str,
        *,
        logprobs: bool,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool | None,
        stop: Any,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": str(prompt)}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self._top_logprobs
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        if stop is not None:
            if isinstance(stop, (list, tuple)):
                payload["stop"] = [str(s) for s in stop if s is not None]
            else:
                payload["stop"] = [str(stop)]
        for key, val in extra.items():
            if key not in payload and val is not None:
                payload[key] = val
        return payload

    def generate(
        self,
        prompt: str,
        *,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        enable_thinking = kwargs.pop("enable_thinking", None)
        stop = kwargs.pop("stop", None)
        extra = {
            k: v
            for k, v in kwargs.items()
            if k not in ("prompt", "logprobs", "max_tokens", "temperature")
        }
        payload = self._build_payload(
            prompt,
            logprobs=logprobs,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking if isinstance(enable_thinking, bool) else None,
            stop=stop,
            extra=extra,
        )
        data = self._post_chat(payload)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = str(message.get("content") or "")
        raw_lp = choice.get("logprobs")
        lp_list = normalize_chat_completion_logprobs(raw_lp) if logprobs else None
        return text, lp_list

    def generate_many(
        self,
        prompt: str,
        *,
        n: int,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> list[tuple[str, list[dict[str, Any]] | None]]:
        nn = max(1, int(n))
        out: list[tuple[str, list[dict[str, Any]] | None]] = []
        for _ in range(nn):
            out.append(
                self.generate(
                    prompt,
                    logprobs=logprobs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            )
        return out


def response_has_thinking_block(text: str) -> bool:
    """True when server output contains Qwen-style thinking delimiters."""
    if not text:
        return False
    return any(p.search(text) for p in _THINKING_MARKERS)


def verify_enable_thinking(
    backend: ServerBackend,
    *,
    probe_prompt: str | None = None,
    max_tokens: int = 128,
) -> tuple[bool, str]:
    """
    Blocker check: ``enable_thinking=True`` must produce thinking markers in response.

    Returns ``(ok, detail)``.
    """
    prompt = probe_prompt or (
        "You are in a text adventure. Exits: north.\n"
        "Think step by step inside the thinking block, then reply with exactly one "
        "game command on a single line (example: go north)."
    )
    try:
        text, _ = backend.generate(
            prompt,
            logprobs=False,
            max_tokens=max_tokens,
            temperature=0.5,
            enable_thinking=True,
        )
    except Exception as exc:
        return False, f"enable_thinking probe request failed: {exc}"
    if response_has_thinking_block(text):
        return True, "thinking markers present"
    return False, "enable_thinking=True but no thinking block in response"


def create_server_backend_from_config(config: dict[str, Any]) -> ServerBackend:
    exec_cfg = config.get("execution") or {}
    inf = config.get("inference") or {}
    model_cfg = config.get("model") or {}
    model_name = str(model_cfg.get("name") or "")
    if not model_name:
        raise BackendError("model.name is required for ServerBackend")
    server_url = str(exec_cfg.get("server_url", "http://127.0.0.1:8000/v1"))
    top_k = resolve_top_logprobs(inf)
    return ServerBackend(
        server_url=server_url,
        model_name=model_name,
        top_logprobs=top_k,
    )
