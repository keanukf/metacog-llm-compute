"""Logprob normalization helpers shared across backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _token_str_from_logprob_obj(obj: Any) -> str | None:
    for attr in ("decoded_token", "token"):
        if hasattr(obj, attr):
            tv = getattr(obj, attr)
            if isinstance(tv, str) and tv:
                return tv
    if isinstance(obj, dict):
        tok = obj.get("token")
        if isinstance(tok, str) and tok:
            return tok
    return None


def _logprob_float_from_obj(obj: Any) -> float | None:
    if hasattr(obj, "logprob"):
        return float(obj.logprob)
    if isinstance(obj, dict):
        lp = obj.get("logprob", obj.get("logprob_value"))
        if lp is not None:
            return float(lp)
    return None


def _candidate_from_entry(token_id: Any, obj: Any) -> dict[str, Any] | None:
    lp = _logprob_float_from_obj(obj)
    if lp is None:
        return None
    cand: dict[str, Any] = {"logprob": lp}
    tok = _token_str_from_logprob_obj(obj)
    if tok:
        cand["token"] = tok
    elif isinstance(token_id, int):
        cand["token_id"] = token_id
    return cand


def _normalize_vllm_position_dict(
    position: dict[Any, Any],
    *,
    sampled_token_id: int | None,
) -> dict[str, Any] | None:
    """Map vLLM ``{token_id: Logprob, ...}`` to canonical token record."""
    candidates: list[dict[str, Any]] = []
    by_id: dict[int, dict[str, Any]] = {}
    for tid, obj in position.items():
        if not isinstance(tid, int):
            continue
        cand = _candidate_from_entry(tid, obj)
        if cand is None:
            continue
        candidates.append(cand)
        by_id[tid] = cand
    if not candidates:
        return None
    candidates.sort(key=lambda c: float(c["logprob"]), reverse=True)

    chosen = by_id.get(sampled_token_id) if sampled_token_id is not None else None
    if chosen is None:
        chosen = candidates[0]

    rec: dict[str, Any] = {
        "logprob": float(chosen["logprob"]),
        "top_logprobs": [
            {"token": c.get("token", ""), "logprob": c["logprob"]} for c in candidates
        ],
    }
    tok = chosen.get("token")
    if isinstance(tok, str) and tok:
        rec["token"] = tok
    return rec


def normalize_logprobs(
    raw: Any,
    *,
    sampled_token_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]] | None:
    """
      Convert backend logprob output to canonical per-token records.

      Each record has ``token``, ``logprob``, and optionally ``top_logprobs`` (list of
    dicts with ``token`` + ``logprob``) for Shannon TLE.
    """
    if raw is None:
        return None
    if not hasattr(raw, "__iter__") or isinstance(raw, (str, bytes)):
        return None
    out: list[dict[str, Any]] = []
    for idx, x in enumerate(raw):
        if x is None:
            continue
        sampled_id: int | None = None
        if sampled_token_ids is not None and idx < len(sampled_token_ids):
            try:
                sampled_id = int(sampled_token_ids[idx])
            except (TypeError, ValueError):
                sampled_id = None

        if isinstance(x, dict):
            lp_val = x.get("logprob", x.get("logprob_value"))
            if lp_val is not None:
                rec: dict[str, Any] = {"logprob": float(lp_val)}
                tok = x.get("token")
                if isinstance(tok, str) and tok:
                    rec["token"] = tok
                top = x.get("top_logprobs")
                if isinstance(top, list) and top:
                    rec["top_logprobs"] = [
                        {
                            "token": str(t.get("token", "")),
                            "logprob": float(t["logprob"]),
                        }
                        for t in top
                        if isinstance(t, dict) and t.get("logprob") is not None
                    ]
                out.append(rec)
                continue

            vals = list(x.values())
            if vals and all(isinstance(k, int) for k in x.keys()):
                vllm_rec = _normalize_vllm_position_dict(x, sampled_token_id=sampled_id)
                if vllm_rec is not None:
                    out.append(vllm_rec)
                    continue

            if not vals:
                continue
            v0 = vals[0]
            if hasattr(v0, "logprob"):
                rec2: dict[str, Any] = {"logprob": float(getattr(v0, "logprob"))}
                tv = _token_str_from_logprob_obj(v0)
                if tv:
                    rec2["token"] = tv
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
            tv = _token_str_from_logprob_obj(x)
            if tv:
                rec4["token"] = tv
            out.append(rec4)
        elif isinstance(x, (int, float)):
            out.append({"logprob": float(x)})
    return out if out else None


def logprob_token_coverage(records: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Summarize how many normalized logprob records include a ``token`` string."""
    if not records:
        return {
            "n_tokens": 0,
            "n_with_token": 0,
            "token_field_rate": 0.0,
            "first_record": None,
        }
    n = len(records)
    n_tok = sum(
        1 for r in records if isinstance(r, dict) and isinstance(r.get("token"), str) and r["token"]
    )
    first = records[0] if records else None
    return {
        "n_tokens": n,
        "n_with_token": n_tok,
        "token_field_rate": float(n_tok) / float(n) if n else 0.0,
        "first_record": first,
    }


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
