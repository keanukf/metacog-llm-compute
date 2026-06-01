"""Logprob normalization helpers shared across backends."""

from __future__ import annotations

from typing import Any


def normalize_logprobs(raw: Any) -> list[dict[str, Any]] | None:
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
                rec: dict[str, Any] = {"logprob": float(lp)}
                tok = x.get("token")
                if isinstance(tok, str) and tok:
                    rec["token"] = tok
                out.append(rec)
                continue

            vals = list(x.values())
            if not vals:
                continue
            v0 = vals[0]
            if hasattr(v0, "logprob"):
                rec2: dict[str, Any] = {"logprob": float(getattr(v0, "logprob"))}
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


def openai_completion_logprobs_to_list(raw_lp: Any) -> list[dict[str, Any]] | None:
    """Map OpenAI Completions choice.logprobs to internal token records."""
    if raw_lp is None:
        return None
    content = getattr(raw_lp, "content", None)
    if content is None and isinstance(raw_lp, dict):
        content = raw_lp.get("content")
    if content is not None:
        return normalize_logprobs(content)

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
