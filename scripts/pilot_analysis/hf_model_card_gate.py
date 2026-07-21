#!/usr/bin/env python3
"""
Hugging Face Hub "model card gate" for thesis model selection.

Fetches public Hub JSON for one or more model repo ids and prints a compact
compliance summary (dense vs MoE-ish signals, instruct tags, thinking keywords,
gating, pipeline tag).

This is intentionally dependency-light (stdlib only) so it works on a fresh
RunPod pod before installing the full experiment stack.

Examples::

  python scripts/pilot_analysis/hf_model_card_gate.py --repo-id Qwen/Qwen3-8B
  python scripts/pilot_analysis/hf_model_card_gate.py --repo-id Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct
  python scripts/pilot_analysis/hf_model_card_gate.py --models-file my_candidates.yaml

``--models-file`` expects an ad-hoc, user-supplied YAML file (a bare list of repo-id strings, or a
``{models: [...]}`` mapping). The shipped shortlist files it used to default to
(``configs/models_runpod.yaml`` / ``configs/models.yaml``) were removed in the 2026-07-21 refactor,
so pass ``--repo-id`` for the single production model or point ``--models-file`` at your own list.

Auth::

  export HF_TOKEN=...  # optional; improves rate limits and enables gated model metadata in some cases
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _http_get_json(url: str, *, token: str | None) -> dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": "metacog-llm-compute/hf_model_card_gate"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted HF API)
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object from {url}, got {type(data).__name__}")
    return data


def _http_get_text(url: str, *, token: str | None) -> str:
    headers = {"Accept": "text/plain", "User-Agent": "metacog-llm-compute/hf_model_card_gate"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _load_models_yaml(path: Path) -> list[str]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError("PyYAML is required for --models-file (install pyyaml)") from e
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, dict):
        m = raw.get("models")
        if isinstance(m, list):
            return [str(x).strip() for x in m if str(x).strip()]
    raise ValueError(f"expected list or {{models: [...]}} in {path}")


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _contains_any(hay: str, needles: list[str]) -> list[str]:
    h = _norm_text(hay)
    hits: list[str] = []
    for n in needles:
        if n.lower() in h:
            hits.append(n)
    return hits


def summarize_repo(repo_id: str, *, token: str | None) -> dict[str, Any]:
    api = f"https://huggingface.co/api/models/{repo_id}"
    meta = _http_get_json(api, token=token)

    tags = meta.get("tags") or []
    tags_l = [str(t) for t in tags if isinstance(t, str)]

    card_url = f"https://huggingface.co/{repo_id}/resolve/main/README.md"
    card_text = ""
    try:
        card_text = _http_get_text(card_url, token=token)
    except urllib.error.HTTPError:
        # Some repos use README.md in a subdir or different branch; keep empty.
        card_text = ""

    config_url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    config_obj: dict[str, Any] | None = None
    try:
        raw_cfg = _http_get_text(config_url, token=token)
        loaded = json.loads(raw_cfg)
        if isinstance(loaded, dict):
            config_obj = loaded
    except Exception:
        config_obj = None

    # IMPORTANT: keep keyword scans mostly on the model card/README.
    # Including the full Hub JSON blob creates lots of false positives (e.g. the word
    # "multimodal" appearing in unrelated citation text).
    readme_blob = card_text[:200_000]

    pipeline_tag = meta.get("pipeline_tag")
    gated = bool(meta.get("gated", False))
    private = bool(meta.get("private", False))

    # Heuristic flags (not a formal architecture parser).
    moe_hits: list[str] = []
    moe_struct: dict[str, Any] = {}
    if config_obj is not None:
        # Common MoE-ish config keys across HF model families (best-effort).
        for k in (
            "num_experts",
            "num_experts_per_tok",
            "moe_layer_freq",
            "moe_intermediate_size",
            "router_aux_loss_coef",
            "moe",
        ):
            if k in config_obj:
                moe_struct[k] = config_obj.get(k)
        archs = config_obj.get("architectures")
        if isinstance(archs, list) and any(
            isinstance(a, str) and "mixtral" in a.lower() for a in archs
        ):
            moe_struct["architectures_hint"] = archs

    if moe_struct:
        moe_hits.append("config.json suggests MoE/expert routing fields present")
    else:
        moe_hits = _contains_any(
            readme_blob,
            [
                "mixture-of-experts",
                "mixture of experts",
                "mixture of expert",
                "mixtral",
                "num_experts",
                "n_experts",
                "experts_per_tok",
                "experts per token",
            ],
        )
        # Avoid generic "moe" substring hits from READMEs that discuss unrelated models.

    # "router" is too generic (appears in unrelated docs); only count it with MoE context.
    if (
        not moe_struct
        and "router" in _norm_text(readme_blob)
        and ("expert" in _norm_text(readme_blob) or "mixtral" in _norm_text(readme_blob))
    ):
        moe_hits.append("router (near-expert/moe language)")

    hybrid_hits = _contains_any(
        readme_blob,
        [
            "deltanet",
            "delta net",
            "hybrid attention",
            "linear attention",
            "mamba",
            "ssd",
            "rwkv",
        ],
    )
    thinking_hits = _contains_any(
        readme_blob,
        [
            "thinking mode",
            "enable_thinking",
            "chain-of-thought",
            "cot mode",
        ],
    )
    # Avoid matching random "/think" substrings in unrelated URLs; require whitespace boundaries.
    if re.search(r"(?i)(^|[\s\"'`])\/think([\s\"'`]|$)", readme_blob):
        thinking_hits.append("/think")

    instruct_hits = _contains_any(
        readme_blob,
        [
            "instruct",
            "chat template",
            "chat_template",
            "instruction tuned",
            "supervised fine-tuning",
            "sft",
        ],
    )

    vl_hits = _contains_any(
        readme_blob, ["vision-language", "image input", "pix2struct", "vl-", "vlm"]
    )
    # "multimodal" is noisy; treat it as VL signal only with extra vision cues.
    if "multimodal" in _norm_text(readme_blob) and (
        "image" in _norm_text(readme_blob)
        or "vision" in _norm_text(readme_blob)
        or "video" in _norm_text(readme_blob)
    ):
        vl_hits.append("multimodal (near vision/image language)")

    verdict_moe = "unknown"
    if moe_struct:
        verdict_moe = "likely_moe_or_expert_routing_per_config_json"
    elif moe_hits:
        verdict_moe = "review_readme_hits_for_moe_language"

    verdict_hybrid = (
        "unknown" if not hybrid_hits else "review_readme_hits_for_nonstandard_attn_language"
    )
    verdict_vl = "unknown" if not vl_hits else "review_readme_hits_for_vl_language"

    return {
        "repo_id": repo_id,
        "pipeline_tag": pipeline_tag,
        "library_name": meta.get("library_name"),
        "likes": meta.get("likes"),
        "downloads": meta.get("downloads"),
        "gated": gated,
        "private": private,
        "tags": tags_l[:40],
        "tag_hits": {
            "looks_vl": [
                t
                for t in tags_l
                if any(x in t.lower() for x in ("vl", "vision", "image", "multimodal"))
            ][:20],
            "looks_moe": [t for t in tags_l if "mixtral" in t.lower() or "moe" in t.lower()][:20],
        },
        "keyword_hits": {
            "moe": moe_hits[:20],
            "hybrid_or_nonstandard_attn": hybrid_hits[:20],
            "thinking": thinking_hits[:20],
            "instruct_chat": instruct_hits[:20],
            "vision_language": vl_hits[:20],
        },
        "heuristic_verdicts": {
            "moe": verdict_moe,
            "hybrid": verdict_hybrid,
            "vision_language": verdict_vl,
        },
        "config_json_url": config_url,
        "config_json_fetched": config_obj is not None,
        "config_moe_fields": moe_struct,
        "card_readme_url": card_url,
        "card_readme_fetched": bool(card_text.strip()),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="HF Hub model-card gate (dense instruct sanity checks)."
    )
    p.add_argument(
        "--repo-id",
        action="append",
        default=[],
        metavar="ORG/NAME",
        help="Model repo id (repeatable)",
    )
    p.add_argument(
        "--models-file",
        type=Path,
        default=None,
        help="Ad-hoc user-supplied YAML: a bare list of repo ids, or {models: [...]}",
    )
    p.add_argument("--json", action="store_true", help="Print one JSON object per line (NDJSON)")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    ids: list[str] = []
    ids.extend([x.strip() for x in (args.repo_id or []) if str(x).strip()])
    if args.models_file is not None:
        path = (
            args.models_file if args.models_file.is_absolute() else (REPO_ROOT / args.models_file)
        )
        ids.extend(_load_models_yaml(path))

    # de-dupe preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for rid in ids:
        if rid not in seen:
            seen.add(rid)
            uniq.append(rid)

    if not uniq:
        p.error("provide --repo-id and/or --models-file")

    rows: list[dict[str, Any]] = []
    for rid in uniq:
        try:
            rows.append(summarize_repo(rid, token=token))
        except urllib.error.HTTPError as e:
            rows.append({"repo_id": rid, "error": f"HTTP {e.code}: {e.reason}"})
        except Exception as e:
            rows.append({"repo_id": rid, "error": f"{type(e).__name__}: {e}"})

    if args.json:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        return

    for r in rows:
        if "error" in r:
            print(f"{r['repo_id']}: ERROR {r['error']}")
            continue
        print(f"== {r['repo_id']} ==")
        print(
            f"pipeline_tag={r.get('pipeline_tag')} library={r.get('library_name')} gated={r.get('gated')} private={r.get('private')}"
        )
        print(
            f"readme_fetched={r.get('card_readme_fetched')} readme_url={r.get('card_readme_url')}"
        )
        kh = r.get("keyword_hits") or {}
        print("keyword_hits:")
        for k, v in kh.items():
            if v:
                print(f"  - {k}: {', '.join(v[:10])}{' …' if len(v) > 10 else ''}")
        th = r.get("tag_hits") or {}
        if any(th.values()):
            print("tag_hits:")
            for k, v in th.items():
                if v:
                    print(f"  - {k}: {', '.join(v[:20])}{' …' if len(v) > 20 else ''}")
        hv = r.get("heuristic_verdicts") or {}
        print("heuristic_verdicts:")
        for k, v in hv.items():
            print(f"  - {k}: {v}")
        tags = r.get("tags") or []
        if tags:
            print("tags (trunc):")
            print("  " + ", ".join(tags[:25]) + (" …" if len(tags) > 25 else ""))
        print()


if __name__ == "__main__":
    main()
