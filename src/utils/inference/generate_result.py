"""Backward-compatible carrier for a backend's ``generate()`` return value.

Every call site and test mock does ``text, logprobs = model.generate(...)`` against a bare
``(text, logprobs)`` 2-tuple -- that contract is frozen (dozens of test mocks rely on it).
``GenerateResult`` unpacks and indexes identically to that 2-tuple (``__iter__``/``__len__``/
``__getitem__`` all yield exactly ``(text, logprobs)``), so existing call sites are unaffected,
but it also carries ``.prompt_tokens`` (the backend-reported input-token count) for call sites
that opt in via ``getattr(result, "prompt_tokens", None)``. A plain tuple (what every test mock
still returns) has no such attribute, so ``getattr`` falls back to ``None`` there -- prompt-token
tracking is therefore backend-real-only by construction, never fabricated for a mock.
"""

from __future__ import annotations

from typing import Any


class GenerateResult:
    __slots__ = ("text", "logprobs", "prompt_tokens")

    def __init__(
        self,
        text: str,
        logprobs: list[dict[str, Any]] | None,
        prompt_tokens: int | None = None,
    ) -> None:
        self.text = text
        self.logprobs = logprobs
        self.prompt_tokens = prompt_tokens

    def __iter__(self):
        return iter((self.text, self.logprobs))

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i: int):
        return (self.text, self.logprobs)[i]

    def __repr__(self) -> str:
        return f"GenerateResult(text={self.text!r}, logprobs=..., prompt_tokens={self.prompt_tokens!r})"
