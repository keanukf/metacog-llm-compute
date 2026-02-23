"""
vLLM wrapper with fallback to HuggingFace Transformers.
Abstract interface: generate(prompt, logprobs=False) -> text, optional logprobs.
"""
from __future__ import annotations

from typing import Any


def _normalize_logprobs(raw: Any) -> list[dict[str, Any]] | None:
    """Convert vLLM/HF logprob output to list of dicts with 'logprob' key."""
    if raw is None:
        return None
    out: list[dict[str, Any]] = []
    if not hasattr(raw, "__iter__") or isinstance(raw, (str, bytes)):
        return None
    for x in raw:
        if isinstance(x, dict):
            lp = x.get("logprob", x.get("logprob_value"))
            if lp is not None:
                out.append({"logprob": float(lp)})
        elif hasattr(x, "logprob"):
            out.append({"logprob": float(x.logprob)})
        elif isinstance(x, (int, float)):
            out.append({"logprob": float(x)})
    return out if out else None


class ModelWrapper:
    """
    Minimal interface for LM inference.
    - generate(prompt, logprobs=False) returns (text, logprobs_or_none).
    - vLLM supports logprobs natively; HF Transformers use output_scores=True.
    """

    def generate(
        self,
        prompt: str,
        *,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """
        Generate a completion for the given prompt.

        Returns:
            (generated_text, logprobs_or_none). logprobs is a list of token-level
            logprob dicts when logprobs=True, else None.
        """
        raise NotImplementedError("Use VLLMWrapper or HFWrapper in production")


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
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._dtype = dtype
        self._max_model_len = max_model_len
        self._kwargs = kwargs
        self._llm: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("VLLMWrapper requires CUDA")
        self._llm = LLM(
            model=self._model_name,
            trust_remote_code=True,
            dtype=self._dtype,
            max_model_len=self._max_model_len,
            **self._kwargs,
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
        from vllm import SamplingParams

        self._ensure_loaded()
        # logprobs=1 returns the chosen token's logprob per position
        logprobs_param = 1 if logprobs else None
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs_param,
            **{k: v for k, v in kwargs.items() if k not in ("prompt", "logprobs", "max_tokens", "temperature")},
        )
        outputs = self._llm.generate([prompt], sampling_params)
        if not outputs or not outputs[0].outputs:
            return "", None
        out = outputs[0].outputs[0]
        text = out.text or ""
        raw_lp = getattr(out, "logprobs", None)
        if raw_lp is None and hasattr(out, "cumulative_logprob"):
            # Some vLLM versions expose cumulative only; we cannot get per-token without logprobs
            lp_list = None
        else:
            lp_list = _normalize_logprobs(raw_lp) if logprobs else None
        return text, lp_list


class HFWrapper(ModelWrapper):
    """
    HuggingFace Transformers-backed model. Fallback when vLLM is unavailable.
    Uses output_scores=True to get logprobs.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        device_map: str = "auto",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._dtype = dtype
        self._device_map = device_map
        self._device = device  # e.g. "mps" for Apple Silicon, "cuda", "cpu"
        self._kwargs = kwargs
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        # On Apple Silicon, device_map="auto" often stays on CPU; use device_map="cpu" then .to("mps")
        device_map = self._device_map
        if self._device == "mps":
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                self._device = "cpu"
            device_map = "cpu"  # load to CPU first, then move to MPS
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._dtype in ("float16", "fp16") else torch.float32,
            device_map=device_map,
            **self._kwargs,
        )
        if self._device and self._device != "cpu":
            try:
                model = model.to(self._device)
            except Exception:
                self._device = "cpu"
        self._model = model

    def generate(
        self,
        prompt: str,
        *,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        import torch

        self._ensure_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 1e-7,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if logprobs:
            gen_kw["output_scores"] = True
            gen_kw["return_dict_in_generate"] = True
        gen_kw.update(kwargs)

        generated = self._model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **gen_kw,
        )
        if logprobs and hasattr(generated, "sequences"):
            # sequences: (1, seq_len); scores: tuple of (1, vocab_size) per generated token
            seq = generated.sequences[0]
            input_len = inputs["input_ids"].shape[1]
            out_ids = seq[input_len:]
            scores = getattr(generated, "scores", None)
            if scores:
                # scores[i] is logits for position input_len + i
                lp_list = []
                for i, s in enumerate(scores):
                    if i >= len(out_ids):
                        break
                    logits = s[0].float()
                    log_probs = torch.log_softmax(logits, dim=-1)
                    tok_id = out_ids[i].item()
                    lp_list.append({"logprob": log_probs[tok_id].item()})
                lp_out = lp_list if lp_list else None
            else:
                lp_out = None
            text = self._tokenizer.decode(out_ids, skip_special_tokens=True)
            return text, lp_out
        else:
            if hasattr(generated, "sequences"):
                out_ids = generated.sequences[0][inputs["input_ids"].shape[1]:]
                text = self._tokenizer.decode(out_ids, skip_special_tokens=True)
            else:
                text = self._tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return text, None


class LiteLLMWrapper(ModelWrapper):
    """
    LiteLLM proxy / OpenAI-compatible API. No local GPU; calls base_url (e.g. http://litellm.home/).
    API key: pass api_key, or set LITELLM_API_KEY in the environment (LiteLLM often expects keys starting with sk-).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://litellm.home/",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._kwargs = kwargs
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "LiteLLMWrapper requires the openai package. Install with: pip install openai"
            ) from e
        import os
        key = self._api_key or os.environ.get("LITELLM_API_KEY") or "dummy"
        self._client = OpenAI(base_url=f"{self._base_url}/v1", api_key=key)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        logprobs: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        client = self._ensure_client()
        extra = {k: v for k, v in kwargs.items() if k not in ("prompt", "logprobs", "max_tokens", "temperature")}
        resp = client.completions.create(
            model=self._model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=1 if logprobs else None,
            **extra,
        )
        if not resp.choices:
            return "", None
        choice = resp.choices[0]
        text = (choice.text or "").strip()
        raw_lp = getattr(choice, "logprobs", None)
        lp_list = _normalize_logprobs(
            getattr(raw_lp, "content", None) if raw_lp else None
        ) if logprobs else None
        return text, lp_list


def create_wrapper(
    backend: str = "vllm",
    model_name: str | None = None,
    dtype: str = "float16",
    device: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> ModelWrapper:
    """
    Factory: create wrapper by backend name.
    - backend "vllm" + model_name -> VLLMWrapper (requires CUDA).
    - backend "hf" + model_name -> HFWrapper; use device="mps" for Apple Silicon, "cuda" or None for auto.
    - backend "litellm" + model_name -> LiteLLMWrapper (OpenAI-compatible API at base_url).
    - Otherwise returns a base ModelWrapper (will raise on generate); use for mocks in tests.
    """
    if model_name and backend == "vllm":
        return VLLMWrapper(model_name=model_name, dtype=dtype, **kwargs)
    if model_name and backend == "hf":
        return HFWrapper(model_name=model_name, dtype=dtype, device=device, **kwargs)
    if model_name and backend == "litellm":
        url = base_url or kwargs.get("litellm_base_url") or "http://litellm.home/"
        api_key = kwargs.get("litellm_api_key") or kwargs.get("api_key")
        rest = {k: v for k, v in kwargs.items() if k not in ("litellm_base_url", "litellm_api_key", "api_key")}
        return LiteLLMWrapper(model_name=model_name, base_url=url, api_key=api_key, **rest)
    # Stub for tests / no model configured
    return ModelWrapper()
