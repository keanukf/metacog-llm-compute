"""vLLM-backed inference wrapper."""

from __future__ import annotations

from typing import Any

from src.utils.errors import BackendError
from src.utils.inference.base import ModelWrapper
from src.utils.inference.logprob_invariance import (
    TLE_INVARIANCE_EPS_BITS,
    capability_probe_prompt,
    probe_temperature_invariance,
)
from src.utils.inference.logprobs import normalize_logprobs

# TLE assumes temperature-invariant logprobs (as if T=1.0). vLLM V1 default is raw_logprobs;
# we pin this explicitly so C2 diversity temperature cannot shift the entropy scale.
_VLLM_LOGPROBS_MODE = "raw_logprobs"


class VLLMWrapper(ModelWrapper):
    """
    vLLM-backed model. Loads model once; generate() runs inference.
    Requires vllm package and CUDA for real use.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        max_model_len: int | None = None,
        chat_template: bool = True,
        enable_thinking: bool = False,
        revision: str | None = None,
        top_logprobs: int = 20,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._dtype = dtype
        self._max_model_len = max_model_len
        self._chat_template = bool(chat_template)
        self._enable_thinking = bool(enable_thinking)
        self._revision = revision
        self._top_logprobs = max(1, int(top_logprobs))
        self._kwargs = kwargs
        self._llm: Any = None
        self._tokenizer: Any = None
        self._logprob_invariance_verified = False

    @property
    def logprobs_mode(self) -> str:
        """Logprobs mode requested for TLE (temperature-invariant raw distribution)."""
        return _VLLM_LOGPROBS_MODE

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        if not torch.cuda.is_available():
            raise BackendError("VLLMWrapper requires CUDA")
        tok_kw: dict[str, Any] = {"trust_remote_code": True}
        llm_kw: dict[str, Any] = {"trust_remote_code": True}
        if self._revision:
            tok_kw["revision"] = self._revision
            llm_kw["revision"] = self._revision
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, **tok_kw)
        llm_kw["logprobs_mode"] = _VLLM_LOGPROBS_MODE
        engine_kwargs = dict(self._kwargs)
        try:
            self._llm = LLM(
                model=self._model_name,
                dtype=self._dtype,
                max_model_len=self._max_model_len,
                **llm_kw,
                **engine_kwargs,
            )
        except TypeError as exc:
            raise BackendError(
                "vLLM LLM() must accept logprobs_mode="
                f"{_VLLM_LOGPROBS_MODE!r} as an engine argument (see vLLM engine args)"
            ) from exc
        self._verify_logprob_invariance_capability()

    def _verify_logprob_invariance_capability(self) -> None:
        """One-shot startup probe: raw logprobs must be invariant to sampling temperature."""
        if self._logprob_invariance_verified:
            return
        diag = probe_temperature_invariance(
            self,
            capability_probe_prompt(),
            t_low=0.3,
            t_high=1.0,
            max_tokens=4,
        )
        cross = diag.get("cross_t_dtle")
        same = diag.get("same_t_dtle")
        if cross is None:
            raise BackendError(
                "vLLM logprob capability probe failed: missing first-token TLE at T=0.3/T=1.0"
            )
        if cross > TLE_INVARIANCE_EPS_BITS:
            raise BackendError(
                "vLLM logprob capability probe failed: first-token |dTLE(0.3 vs 1.0)|="
                f"{cross:.4f} bits exceeds eps={TLE_INVARIANCE_EPS_BITS} "
                f"(same-T noise floor={same}); logprobs may not be raw_logprobs"
            )
        self._logprob_invariance_verified = True

    def _maybe_apply_chat_template(
        self, prompt: str, *, enable_thinking: bool | None = None
    ) -> str:
        if not self._chat_template:
            return prompt
        tok = self._tokenizer
        if tok is None or not hasattr(tok, "apply_chat_template"):
            return prompt
        messages = [{"role": "user", "content": str(prompt)}]
        use_thinking = self._enable_thinking if enable_thinking is None else bool(enable_thinking)
        try:
            rendered = tok.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=use_thinking,
            )
        except TypeError:
            rendered = tok.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return rendered if isinstance(rendered, str) and rendered else prompt

    def _default_stop_tokens(self) -> list[str]:
        out: list[str] = []
        tok = self._tokenizer
        if tok is None:
            return out
        eos = getattr(tok, "eos_token", None)
        if isinstance(eos, str) and eos:
            out.append(eos)
        out.append("<|im_end|>")
        return out

    def _merge_stop(self, user_stop: Any) -> list[str] | None:
        default_stop = self._default_stop_tokens()
        if user_stop is None:
            return default_stop or None
        merged: list[str] = []
        if isinstance(user_stop, (list, tuple)):
            merged.extend(str(s) for s in user_stop if s is not None)
        else:
            merged.append(str(user_stop))
        merged.extend(s for s in default_stop if s and s not in merged)
        return merged or None

    def _sampling_params(
        self,
        *,
        temperature: float,
        max_tokens: int,
        logprobs: bool,
        merged_stop: list[str] | None,
        extra: dict[str, Any],
        n: int | None = None,
    ) -> Any:
        from vllm import SamplingParams

        base: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": self._top_logprobs if logprobs else None,
            "stop": merged_stop,
            **extra,
        }
        if n is not None:
            base["n"] = int(n)
        return SamplingParams(**base)

    def generate(
        self,
        prompt: str,
        *,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        self._ensure_loaded()
        et = kwargs.pop("enable_thinking", None)
        rendered_prompt = self._maybe_apply_chat_template(prompt, enable_thinking=et)
        merged_stop = self._merge_stop(kwargs.get("stop"))

        extra = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "prompt",
                "logprobs",
                "max_tokens",
                "temperature",
                "stop",
                "enable_thinking",
            )
        }
        sampling_params = self._sampling_params(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            merged_stop=merged_stop,
            extra=extra,
        )
        outputs = self._llm.generate([rendered_prompt], sampling_params)
        if not outputs or not outputs[0].outputs:
            return "", None
        out = outputs[0].outputs[0]
        text = out.text or ""
        raw_lp = getattr(out, "logprobs", None)
        token_ids = getattr(out, "token_ids", None)
        if raw_lp is None and hasattr(out, "cumulative_logprob"):
            lp_list = None
        else:
            lp_list = normalize_logprobs(raw_lp, sampled_token_ids=token_ids) if logprobs else None
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
        try:
            from vllm import SamplingParams  # noqa: F401
        except Exception:
            return super().generate_many(
                prompt,
                n=n,
                logprobs=logprobs,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

        self._ensure_loaded()
        et = kwargs.pop("enable_thinking", None)
        rendered_prompt = self._maybe_apply_chat_template(prompt, enable_thinking=et)
        merged_stop = self._merge_stop(kwargs.get("stop"))

        extra = {
            k: v
            for k, v in kwargs.items()
            if k
            not in ("prompt", "logprobs", "max_tokens", "temperature", "stop", "enable_thinking")
        }
        nn = max(1, int(n))
        try:
            sampling_params = self._sampling_params(
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=logprobs,
                merged_stop=merged_stop,
                extra=extra,
                n=nn,
            )
            outputs = self._llm.generate([rendered_prompt], sampling_params)
            if not outputs or not outputs[0].outputs:
                return []
            out: list[tuple[str, list[dict[str, Any]] | None]] = []
            for o in outputs[0].outputs:
                text = o.text or ""
                raw_lp = getattr(o, "logprobs", None)
                token_ids = getattr(o, "token_ids", None)
                lp_list = (
                    normalize_logprobs(raw_lp, sampled_token_ids=token_ids) if logprobs else None
                )
                out.append((text, lp_list))
            return out
        except TypeError:
            return super().generate_many(
                prompt,
                n=nn,
                logprobs=logprobs,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=merged_stop,
                enable_thinking=et,
            )
