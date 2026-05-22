"""
vLLM wrapper with fallback to HuggingFace Transformers.
Abstract interface: generate(prompt, logprobs=False) -> text, optional logprobs.
"""

from __future__ import annotations

import json
import warnings
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.utils.errors import BackendError


def normalize_openai_base_url(base_url: str) -> str:
    """
    OpenAI Python client expects base_url ending with /v1 (no duplicate segment).
    Accepts either a server root (e.g. http://host:1234) or full API base (…/v1).
    """
    bu = base_url.strip().rstrip("/")
    if bu.endswith("/v1"):
        return bu
    return f"{bu}/v1"


def _normalize_logprobs(raw: Any) -> list[dict[str, Any]] | None:
    """Convert vLLM/HF logprob output to list of dicts with 'logprob' key."""
    if raw is None:
        return None
    out: list[dict[str, Any]] = []
    if not hasattr(raw, "__iter__") or isinstance(raw, (str, bytes)):
        return None
    for x in raw:
        # Common shapes:
        # - list[{"logprob": -0.1, ...}, ...]
        # - list[float, ...]
        # - vLLM v0.19+: list[dict[token_id -> Logprob]] when SamplingParams(logprobs=k)
        if isinstance(x, dict):
            lp = x.get("logprob", x.get("logprob_value"))
            if lp is not None:
                rec: dict[str, Any] = {"logprob": float(lp)}
                tok = x.get("token")
                if isinstance(tok, str) and tok:
                    rec["token"] = tok
                out.append(rec)
                continue

            # vLLM: dict[int, Logprob] (or dict[str, Logprob]) for this generated position.
            # With logprobs=1, this often contains exactly one entry (the chosen token).
            vals = list(x.values())
            if not vals:
                continue
            v0 = vals[0]
            if hasattr(v0, "logprob"):
                rec2: dict[str, Any] = {"logprob": float(getattr(v0, "logprob"))}
                # Best-effort: some vLLM Logprob objects expose decoded_token / token.
                for attr in ("decoded_token", "token"):
                    if hasattr(v0, attr):
                        tv = getattr(v0, attr)
                        if isinstance(tv, str) and tv:
                            rec2["token"] = tv
                            break
                out.append(rec2)
                continue
            if isinstance(v0, dict):
                lp2 = v0.get("logprob", v0.get("logprob_value"))
                if lp2 is not None:
                    rec3: dict[str, Any] = {"logprob": float(lp2)}
                    tok2 = v0.get("token")
                    if isinstance(tok2, str) and tok2:
                        rec3["token"] = tok2
                    out.append(rec3)
                    continue
        elif hasattr(x, "logprob"):
            rec4: dict[str, Any] = {"logprob": float(x.logprob)}
            for attr in ("decoded_token", "token"):
                if hasattr(x, attr):
                    tv = getattr(x, attr)
                    if isinstance(tv, str) and tv:
                        rec4["token"] = tv
                        break
            out.append(rec4)
        elif isinstance(x, (int, float)):
            out.append({"logprob": float(x)})
    return out if out else None


def _openai_completion_logprobs_to_list(raw_lp: Any) -> list[dict[str, Any]] | None:
    """
    Map OpenAI *Completions* choice.logprobs to internal [{"logprob": float}, ...].
    Also accepts chat-style .content token lists (some proxies) and plain dicts from JSON.
    """
    if raw_lp is None:
        return None
    content = getattr(raw_lp, "content", None)
    if content is None and isinstance(raw_lp, dict):
        content = raw_lp.get("content")
    if content is not None:
        return _normalize_logprobs(content)

    token_lps = getattr(raw_lp, "token_logprobs", None)
    if token_lps is None and isinstance(raw_lp, dict):
        token_lps = raw_lp.get("token_logprobs")
    if token_lps is not None:
        out: list[dict[str, Any]] = []
        for lp in token_lps:
            if lp is not None:
                out.append({"logprob": float(lp)})
        return out if out else None
    return None


def _lmstudio_consume_logprobs_list(raw: Any, token_records: list[dict[str, Any]]) -> None:
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


def _lmstudio_extract_reasoning_and_message(
    data: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    """
    Split LM Studio /v1/responses ``output`` into (reasoning_text, message_text, logprobs).

    Qwen3 on LM Studio often returns ``type: reasoning`` (reasoning_text) plus
    ``type: message`` (output_text) instead of inline `` blocks.
    """
    reasoning_chunks: list[str] = []
    message_chunks: list[str] = []
    token_records: list[dict[str, Any]] = []

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
                _lmstudio_consume_logprobs_list(part.get("logprobs"), token_records)

    out_blocks = data.get("output")
    if isinstance(out_blocks, list):
        for block in out_blocks:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type") or "")
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
                _lmstudio_consume_logprobs_list(block.get("logprobs"), token_records)
                continue
            _walk_parts(block.get("content"))

    reasoning_text = "\n\n".join(reasoning_chunks).strip()
    message_text = "".join(message_chunks).strip()
    return reasoning_text, message_text, token_records


def _lmstudio_assemble_assistant_text(
    reasoning_text: str,
    message_text: str,
    *,
    enable_thinking: bool | None,
) -> str:
    """Map LM Studio split blocks to Qwen3-style text our C1 parser understands."""
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

    LM Studio 0.4+ may return separate ``reasoning`` and ``message`` output blocks for Qwen3.
    When ``enable_thinking`` is True (or unset and reasoning is present), reasoning is wrapped
    in `` blocks so :func:`src.agent.cot_parser.parse_cot_action` can use ``post_think``.
    When ``enable_thinking`` is False, only ``message`` output_text is returned (verify/C0).
    """
    if not isinstance(data, dict):
        return "", None
    reasoning_text, message_text, token_records = _lmstudio_extract_reasoning_and_message(data)
    use_thinking = enable_thinking
    if use_thinking is None and reasoning_text:
        use_thinking = True
    text = _lmstudio_assemble_assistant_text(
        reasoning_text,
        message_text,
        enable_thinking=use_thinking,
    )
    if not text and token_records:
        text = "".join(rec.get("token", "") for rec in token_records).strip()
    if not token_records:
        return text, None
    return text, token_records


def _lmstudio_post_v1_responses(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_logprobs: int,
    enable_thinking: bool | None = None,
) -> dict[str, Any] | None:
    """POST JSON to {base}/responses. Returns parsed dict or None on failure."""
    api = normalize_openai_base_url(base_url).rstrip("/")
    url = f"{api}/responses"
    # LM Studio 0.4+ Responses API; field names aligned with OpenAI Responses where possible.
    body: dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "include": ["message.output_text.logprobs"],
        "top_logprobs": int(top_logprobs),
        "temperature": float(temperature),
        "max_output_tokens": int(max_tokens),
    }
    if enable_thinking is not None:
        et = bool(enable_thinking)
        body["enable_thinking"] = et
        # LM Studio Qwen3: chat_template_kwargs is the documented toggle (top-level may be ignored).
        body["chat_template_kwargs"] = {"enable_thinking": et}
    payload = json.dumps(body).encode("utf-8")
    req = Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlopen(req, timeout=600) as resp:
            raw = resp.read().decode("utf-8")
            out = json.loads(raw)
            return out if isinstance(out, dict) else None
    except HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            err_body = ""
        # Retry once with max_tokens if server rejects max_output_tokens
        if e.code == 400 and "max_output_tokens" in body:
            body2 = dict(body)
            body2.pop("max_output_tokens", None)
            body2["max_tokens"] = int(max_tokens)
            try:
                req2 = Request(
                    url,
                    data=json.dumps(body2).encode("utf-8"),
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                with urlopen(req2, timeout=600) as resp2:
                    raw2 = resp2.read().decode("utf-8")
                    out2 = json.loads(raw2)
                    return out2 if isinstance(out2, dict) else None
            except Exception as e2:
                warnings.warn(
                    f"LM Studio POST /v1/responses failed (retry): {e2!s}; first error body: {err_body!r}"
                )
                return None
        warnings.warn(f"LM Studio POST /v1/responses HTTP {e.code}: {err_body!r}")
        return None
    except (URLError, OSError, json.JSONDecodeError, ValueError) as e:
        warnings.warn(f"LM Studio POST /v1/responses failed: {e!s}")
        return None


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
        """
        Generate N independent samples for the same prompt.

        Default implementation falls back to calling ``generate`` N times.
        Backends may override this to exploit native multi-sampling (e.g. vLLM SamplingParams(n=N)).
        """
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
        """
        For instruct/chat-tuned models (e.g. Qwen3), wrap raw text as a single user turn
        and apply the model's chat template so generation starts in the right mode.

        ``enable_thinking``: if set, overrides the wrapper default (e.g. VC follow-up must use False
        even when action calls use ``inference.enable_thinking: true``).
        """
        if not self._chat_template:
            return prompt
        tok = self._tokenizer
        if tok is None or not hasattr(tok, "apply_chat_template"):
            return prompt
        messages = [{"role": "user", "content": str(prompt)}]
        use_thinking = self._enable_thinking if enable_thinking is None else bool(enable_thinking)
        # Qwen3 templates support enable_thinking; older templates may not.
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
        # Common chat boundary token for chat templates.
        out.append("<|im_end|>")
        return out

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
        # logprobs=1 returns the chosen token's logprob per position
        logprobs_param = 1 if logprobs else None
        user_stop = kwargs.get("stop")
        default_stop = self._default_stop_tokens()
        merged_stop: list[str] | None = None
        if user_stop is None:
            merged_stop = default_stop or None
        else:
            merged: list[str] = []
            if isinstance(user_stop, (list, tuple)):
                merged.extend(str(s) for s in user_stop if s is not None)
            else:
                merged.append(str(user_stop))
            merged.extend(s for s in default_stop if s and s not in merged)
            merged_stop = merged or None

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
            # Some vLLM versions expose cumulative only; we cannot get per-token without logprobs
            lp_list = None
        else:
            lp_list = _normalize_logprobs(raw_lp) if logprobs else None
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
        """
        vLLM-native multi-sampling when supported; falls back to loop otherwise.
        """
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
        user_stop = kwargs.get("stop")
        default_stop = self._default_stop_tokens()
        merged_stop: list[str] | None = None
        if user_stop is None:
            merged_stop = default_stop or None
        else:
            merged: list[str] = []
            if isinstance(user_stop, (list, tuple)):
                merged.extend(str(s) for s in user_stop if s is not None)
            else:
                merged.append(str(user_stop))
            merged.extend(s for s in default_stop if s and s not in merged)
            merged_stop = merged or None

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
                lp_list = _normalize_logprobs(raw_lp) if logprobs else None
                out.append((text, lp_list))
            return out
        except TypeError:
            # Some vLLM versions do not support SamplingParams(n=...). Fall back safely.
            return super().generate_many(
                prompt,
                n=nn,
                logprobs=logprobs,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=merged_stop,
                enable_thinking=et,
            )


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
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        # HuggingFace ``generate`` does not accept OpenAI-style ``stop``; agent uses first-line extraction instead.
        hf_kwargs = {k: v for k, v in kwargs.items() if k not in ("stop", "enable_thinking")}
        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 1e-7,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if logprobs:
            gen_kw["output_scores"] = True
            gen_kw["return_dict_in_generate"] = True
        gen_kw.update(hf_kwargs)

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
                    tok_str = self._tokenizer.decode([tok_id], skip_special_tokens=True)
                    rec: dict[str, Any] = {"logprob": log_probs[tok_id].item()}
                    if isinstance(tok_str, str) and tok_str:
                        rec["token"] = tok_str
                    lp_list.append(rec)
                lp_out = lp_list if lp_list else None
            else:
                lp_out = None
            text = self._tokenizer.decode(out_ids, skip_special_tokens=True)
            return text, lp_out
        else:
            if hasattr(generated, "sequences"):
                out_ids = generated.sequences[0][inputs["input_ids"].shape[1] :]
                text = self._tokenizer.decode(out_ids, skip_special_tokens=True)
            else:
                text = self._tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
            return text, None


class LMStudioWrapper(ModelWrapper):
    """
    LM Studio (or compatible) OpenAI HTTP API. No local model load; calls base_url (e.g. http://host:1234/v1).

    - ``logprobs=False``: uses ``/v1/completions`` (OpenAI client).
    - ``logprobs=True``: uses ``POST /v1/responses`` with ``include: message.output_text.logprobs`` (LM Studio 0.4+).
      Per-token records may include ``top_logprobs`` for Shannon TLE in ``token_entropy``.

    API key: pass api_key, or set LM_STUDIO_API_KEY (default placeholder used by LM Studio if unset).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str | None = None,
        *,
        lmstudio_top_logprobs: int = 5,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.strip()
        self._api_key = api_key
        self._top_logprobs = max(1, int(lmstudio_top_logprobs))
        self._kwargs = kwargs
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise BackendError(
                "LMStudioWrapper requires the openai package. Install with: pip install openai"
            ) from e
        import os

        key = self._api_key or os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
        self._client = OpenAI(base_url=normalize_openai_base_url(self._base_url), api_key=key)
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
        import os

        enable_thinking = kwargs.pop("enable_thinking", None)
        extra = {
            k: v
            for k, v in kwargs.items()
            if k not in ("prompt", "logprobs", "max_tokens", "temperature", "enable_thinking")
        }

        thinking_kw: dict[str, Any] = {}
        if isinstance(enable_thinking, bool):
            thinking_kw["enable_thinking"] = bool(enable_thinking)

        if logprobs:
            key = self._api_key or os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
            data = _lmstudio_post_v1_responses(
                base_url=self._base_url,
                api_key=key,
                model=self._model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_logprobs=self._top_logprobs,
                **thinking_kw,
            )
            if data is not None:
                text, lp_list = parse_lmstudio_responses_json(data, enable_thinking=enable_thinking)
                if text or lp_list:
                    if logprobs and not lp_list:
                        warnings.warn(
                            "LM Studio /v1/responses returned text but no logprobs; "
                            "TLE unavailable for this call."
                        )
                    return text, lp_list if logprobs else None
                warnings.warn(
                    "LM Studio /v1/responses returned empty text and logprobs; "
                    "falling back to /v1/completions."
                )

        client = self._ensure_client()
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
        lp_list = _openai_completion_logprobs_to_list(raw_lp) if logprobs else None
        # Do not fabricate fake logprobs when the server returns null (would yield misleading TLE=0).
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
    - backend "lmstudio" + model_name -> LMStudioWrapper (OpenAI-compatible API at base_url).
    - Otherwise returns a base ModelWrapper (will raise on generate); use for mocks in tests.
    """
    if model_name and backend == "vllm":
        return VLLMWrapper(model_name=model_name, dtype=dtype, **kwargs)
    if model_name and backend == "hf":
        return HFWrapper(model_name=model_name, dtype=dtype, device=device, **kwargs)
    if model_name and backend == "lmstudio":
        url = base_url or kwargs.get("lmstudio_base_url") or "http://localhost:1234/v1"
        api_key = kwargs.get("lmstudio_api_key") or kwargs.get("api_key")
        rest = {
            k: v
            for k, v in kwargs.items()
            if k not in ("lmstudio_base_url", "lmstudio_api_key", "api_key")
        }
        top_k = int(rest.pop("lmstudio_top_logprobs", kwargs.get("lmstudio_top_logprobs", 5)))
        return LMStudioWrapper(
            model_name=model_name,
            base_url=url,
            api_key=api_key,
            lmstudio_top_logprobs=top_k,
            **rest,
        )
    # Stub for tests / no model configured
    return ModelWrapper()
