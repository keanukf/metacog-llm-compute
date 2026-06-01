"""vLLM-backed inference wrapper."""

from __future__ import annotations

from typing import Any

from src.utils.errors import BackendError
from src.utils.inference.base import ModelWrapper
from src.utils.inference.logprobs import normalize_logprobs


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
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._dtype = dtype
        self._max_model_len = max_model_len
        self._chat_template = bool(chat_template)
        self._enable_thinking = bool(enable_thinking)
        self._kwargs = kwargs
        self._llm: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        if not torch.cuda.is_available():
            raise BackendError("VLLMWrapper requires CUDA")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        self._llm = LLM(
            model=self._model_name,
            trust_remote_code=True,
            dtype=self._dtype,
            max_model_len=self._max_model_len,
            **self._kwargs,
        )

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

    def generate(
        self,
        prompt: str,
        *,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        from vllm import SamplingParams

        self._ensure_loaded()
        et = kwargs.pop("enable_thinking", None)
        rendered_prompt = self._maybe_apply_chat_template(prompt, enable_thinking=et)
        logprobs_param = 1 if logprobs else None
        merged_stop = self._merge_stop(kwargs.get("stop"))

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs_param,
            stop=merged_stop,
            **{
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
            },
        )
        outputs = self._llm.generate([rendered_prompt], sampling_params)
        if not outputs or not outputs[0].outputs:
            return "", None
        out = outputs[0].outputs[0]
        text = out.text or ""
        raw_lp = getattr(out, "logprobs", None)
        if raw_lp is None and hasattr(out, "cumulative_logprob"):
            lp_list = None
        else:
            lp_list = normalize_logprobs(raw_lp) if logprobs else None
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
            from vllm import SamplingParams
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
        logprobs_param = 1 if logprobs else None
        merged_stop = self._merge_stop(kwargs.get("stop"))

        extra = {
            k: v
            for k, v in kwargs.items()
            if k
            not in ("prompt", "logprobs", "max_tokens", "temperature", "stop", "enable_thinking")
        }
        nn = max(1, int(n))
        try:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=logprobs_param,
                stop=merged_stop,
                n=nn,
                **extra,
            )
            outputs = self._llm.generate([rendered_prompt], sampling_params)
            if not outputs or not outputs[0].outputs:
                return []
            out: list[tuple[str, list[dict[str, Any]] | None]] = []
            for o in outputs[0].outputs:
                text = o.text or ""
                raw_lp = getattr(o, "logprobs", None)
                lp_list = normalize_logprobs(raw_lp) if logprobs else None
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
