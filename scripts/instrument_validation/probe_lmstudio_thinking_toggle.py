#!/usr/bin/env python3
"""
Probe LM Studio ``POST /v1/responses`` thinking controls.

Compares payload variants (template kwargs vs ``reasoning.effort`` none/low/medium)
and prints reasoning vs message token usage. Note: HTTP API uses Open Responses enum;
LM Studio maps to model on/off internally (``on``/``off`` are not valid wire values).

Requires a running LM Studio local server and a loaded Qwen3 (or similar) model.

Example:
  python scripts/probe_lmstudio_thinking_toggle.py --model qwen/qwen3-4b
  python scripts/probe_lmstudio_thinking_toggle.py --config configs/lmstudio_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.inference.lmstudio.request_body import (  # noqa: E402
    build_v1_responses_body,
    summarize_responses_usage,
    thinking_control_fields,
)
from src.utils.inference.lmstudio.responses import (  # noqa: E402
    default_api_key,
    post_v1_responses,
    resolve_lmstudio_api_host,
)
from src.utils.inference.urls import normalize_openai_base_url  # noqa: E402
from src.utils.pilot_config import load_yaml_path  # noqa: E402


def _short_prompt() -> str:
    return (
        "You are in a text adventure. Exits: north.\n"
        "Reply with exactly one game command on a single line (example: go north). "
        "No explanation."
    )


def _variant_bodies(model: str, *, max_tokens: int) -> list[tuple[str, dict[str, Any]]]:
    base = {
        "model": model,
        "input": [{"role": "user", "content": _short_prompt()}],
        "temperature": 0.3,
        "max_output_tokens": max_tokens,
        "include": ["message.output_text.logprobs"],
        "top_logprobs": 5,
    }

    variants: list[tuple[str, dict[str, Any]]] = []

    body_none = dict(base)
    variants.append(("no_thinking_fields", body_none))

    body_ct = dict(base)
    body_ct["enable_thinking"] = False
    body_ct["chat_template_kwargs"] = {"enable_thinking": False}
    variants.append(("chat_template_kwargs_only_off", body_ct))

    body_none = dict(base)
    body_none["reasoning"] = {"effort": "none"}
    variants.append(("reasoning_effort_none_only", body_none))

    body_low = dict(base)
    body_low["reasoning"] = {"effort": "low"}
    variants.append(("reasoning_effort_low_only", body_low))

    body_medium = dict(base)
    body_medium["reasoning"] = {"effort": "medium"}
    variants.append(("reasoning_effort_medium_only", body_medium))

    # Invalid on wire (HTTP 400) — model layer uses on/off in dev log only.
    body_off = dict(base)
    body_off["reasoning"] = {"effort": "off"}
    variants.append(("invalid_wire_off", body_off))

    body_full_off = build_v1_responses_body(
        model=model,
        prompt=_short_prompt(),
        max_tokens=max_tokens,
        temperature=0.3,
        top_logprobs=5,
        enable_thinking=False,
    )
    variants.append(("repo_full_off", body_full_off))

    body_full_on = build_v1_responses_body(
        model=model,
        prompt=_short_prompt(),
        max_tokens=max_tokens,
        temperature=0.3,
        top_logprobs=5,
        enable_thinking=True,
    )
    variants.append(("repo_full_on", body_full_on))

    # Documented workaround when template kwargs are ignored (LM Studio bug tracker #1559).
    body_prefill = build_v1_responses_body(
        model=model,
        prompt=_short_prompt(),
        max_tokens=max_tokens,
        temperature=0.3,
        top_logprobs=5,
        enable_thinking=False,
        extra_input_messages=[{"role": "assistant", "content": " \n"}],
    )
    variants.append(("repo_full_off_assistant_prefill", body_prefill))

    return variants


def _run_variant(
    *,
    base_url: str,
    api_key: str,
    name: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    import httpx

    url = f"{normalize_openai_base_url(base_url).rstrip('/')}/responses"
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
    except httpx.HTTPError as exc:
        return {"variant": name, "error": str(exc)}

    row: dict[str, Any] = {
        "variant": name,
        "http_status": resp.status_code,
        "request_thinking": {
            "enable_thinking": body.get("enable_thinking"),
            "chat_template_kwargs": body.get("chat_template_kwargs"),
            "reasoning": body.get("reasoning"),
        },
    }
    if resp.status_code >= 400:
        row["error"] = (resp.text or "")[:500]
        return row

    try:
        data = resp.json()
    except ValueError as exc:
        row["error"] = f"invalid JSON: {exc}"
        return row

    if isinstance(data, dict):
        row.update(summarize_responses_usage(data))
        row["ok_for_action"] = bool(row.get("has_message_block")) and bool(
            row.get("message_preview")
        )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe LM Studio enable_thinking / reasoning.effort")
    ap.add_argument("--config", default=None, help="Optional YAML with model.name and inference.*")
    ap.add_argument("--model", default=None, help="Model id (overrides config)")
    ap.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default env LM_STUDIO_BASE_URL or localhost:1234/v1)",
    )
    ap.add_argument("--max-tokens", type=int, default=128, help="max_output_tokens per variant")
    ap.add_argument(
        "--json-out",
        default=None,
        help="Write full results JSON to this path",
    )
    args = ap.parse_args()

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_yaml_path(
            REPO_ROOT / args.config if not Path(args.config).is_absolute() else args.config
        )

    model = (
        args.model or (cfg.get("model") or {}).get("name") or os.environ.get("LMSTUDIO_PROBE_MODEL")
    )
    if not model:
        print("error: pass --model or --config with model.name", file=sys.stderr)
        return 2

    inf = cfg.get("inference") if isinstance(cfg.get("inference"), dict) else {}
    base_url = (
        args.base_url
        or inf.get("lmstudio_base_url")
        or os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    )
    api_key = default_api_key(inf.get("lmstudio_api_key"))

    try:
        host = resolve_lmstudio_api_host(base_url)
        print(f"LM Studio api_host: {host}")
    except Exception as exc:
        print(f"warning: could not probe api_host: {exc}")

    print(f"model={model!r} base_url={base_url!r} max_output_tokens={args.max_tokens}")
    print(f"repo thinking_control_fields(False): {thinking_control_fields(False)}")
    print()

    results: list[dict[str, Any]] = []
    for name, body in _variant_bodies(model, max_tokens=args.max_tokens):
        row = _run_variant(base_url=base_url, api_key=api_key, name=name, body=body)
        results.append(row)
        status = row.get("http_status", "?")
        rtok = row.get("reasoning_tokens", "?")
        blocks = row.get("output_block_types", [])
        preview = row.get("message_preview", "")
        ok = row.get("ok_for_action", False)
        print(
            f"{name:36} http={status} reasoning_tokens={rtok} blocks={blocks} "
            f"ok_for_action={ok} preview={preview!r}"
        )
        if row.get("error"):
            print(f"  error: {row['error']}")

    print()
    print("Interpretation:")
    print(
        "  - Wire: reasoning.effort 'none' (off) | 'low' (on, repo default) — not 'on'/'off' (HTTP 400)"
    )
    print(
        "  - Avoid 'medium' on qwen3-4b if dev log warns; repo uses 'low' for enable_thinking=True"
    )
    print(
        "  - reasoning_tokens=0 + message block => thinking OFF; reasoning_only => bad for C0 parser"
    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")

    # Also verify wrapper path uses same body builder
    data, status, err = post_v1_responses(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=_short_prompt(),
        max_tokens=args.max_tokens,
        temperature=0.3,
        top_logprobs=5,
        enable_thinking=False,
    )
    if data is None:
        print(f"\npost_v1_responses(enable_thinking=False) failed: {err} (http {status})")
    else:
        summary = summarize_responses_usage(data)
        print(f"\npost_v1_responses wrapper: {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
