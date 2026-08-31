#!/usr/bin/env python3
"""Gate F run-hygiene preflight (HART) — fast pod-side checks before a real Phase 1/2 block.

Checks the five things blueprints/gate_p1_readiness.md's "Run-Hygiene auf dem Pod" item requires,
all without running any episodes or touching the GPU:

1. Model pre-staged on the *ephemeral* container disk, not the persistent network volume.
   ``HF_HOME`` must live outside ``/workspace`` (RunPod's template default points it *at*
   ``/workspace/.cache``, which is exactly backwards -- see docs/runpod.md and
   scripts/cloud/shell/pod_runtime_env.sh) -- keeping weights off the network volume is deliberate, not an
   oversight; re-download on a fresh container disk is expected and budgeted for. Also verifies
   the frozen model+revision's snapshot is actually present in that cache, so the first real
   inference call doesn't stall on a cold download mid-batch.
2. Langfuse tracing: either explicitly off, or credentials resolved and the SDK importable.
3. The repo checkout itself resolves under ``/workspace`` (the actual persistence mechanism --
   ``--checkpoint-dir`` defaults to a path relative to the repo root, so results only survive a
   pod restart if the repo itself sits on the network volume; the ``RESULTS_DIR`` env var some
   docs mention is not read by any script, it's a convention only, checked here too if set).
4. History-guard: no truncation params active in the resolved step config for either domain
   (mirrors src/utils/history_guard.py's own runtime check, but fails fast here instead of
   mid-batch).
5. Logprob sidecar config: mode == action_window, full-instance overrides are the documented
   non-holdout instances and don't collide with the mod-10 holdout set; execution N matches the
   C-1-verified batch-invariant value.

Usage:
    python scripts/run_readiness/run_hygiene_preflight.py --config configs/experiment_core.yaml

Exit code 0 iff every check passes.
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

# C-1 scoped waiver (docs/consistency_log.md, 2026-07-13) froze concurrency at this value;
# It is a project-wide invariant and is not to be silently changed.
EXPECTED_MAX_CONCURRENT_EPISODES = 32
DOMAINS = ("textworld", "tower_of_hanoi")


def _check(label: str, ok: bool, detail: str) -> dict[str, Any]:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return {"check": label, "pass": ok, "detail": detail}


def check_model_on_ephemeral_disk(config: dict[str, Any]) -> dict[str, Any]:
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        return _check(
            "model_on_ephemeral_disk",
            False,
            "HF_HOME is not set (source scripts/cloud/shell/pod_runtime_env.sh).",
        )
    if hf_home.rstrip("/").startswith("/workspace"):
        return _check(
            "model_on_ephemeral_disk",
            False,
            f"HF_HOME={hf_home} is under /workspace (persistent network volume) -- model weights "
            "must stay on the ephemeral container disk (network volume should hold code+results "
            "only, docs/runpod.md). Source scripts/cloud/shell/pod_runtime_env.sh before setup/inference.",
        )
    model_cfg = config.get("model") or {}
    name = str(model_cfg.get("name", ""))
    revision = str(model_cfg.get("revision", ""))
    if "/" not in name:
        return _check(
            "model_on_ephemeral_disk", False, f"model.name={name!r} is not in 'org/repo' form."
        )
    org, repo = name.split("/", 1)
    snapshot_dir = Path(hf_home) / "hub" / f"models--{org}--{repo}" / "snapshots" / revision
    cached = snapshot_dir.is_dir() and any(snapshot_dir.iterdir())
    return _check(
        "model_on_ephemeral_disk",
        cached,
        f"HF_HOME={hf_home} (ephemeral, correct); {name}@{revision} "
        f"{'cached at ' + str(snapshot_dir) if cached else 'NOT found at ' + str(snapshot_dir) + ' -- pre-download before the real run'}.",
    )


def check_langfuse(config: dict[str, Any]) -> dict[str, Any]:
    tracing_cfg = config.get("tracing") or {}
    if not bool(tracing_cfg.get("langfuse_enabled")):
        return _check(
            "langfuse", True, "tracing.langfuse_enabled=false (explicitly off, documented)."
        )

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY") or tracing_cfg.get("langfuse_public_key")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY") or tracing_cfg.get("langfuse_secret_key")
    creds_ok = bool(public_key) and bool(secret_key)
    try:
        import langfuse  # noqa: F401

        sdk_ok = True
    except ImportError:
        sdk_ok = False
    ok = creds_ok and sdk_ok
    return _check(
        "langfuse",
        ok,
        f"langfuse_enabled=true, credentials_resolved={creds_ok}, sdk_installed={sdk_ok}"
        + ("" if ok else " -- source /workspace/secrets/env.sh or set langfuse_enabled: false"),
    )


def check_repo_on_network_volume() -> dict[str, Any]:
    resolved = REPO_ROOT.resolve()
    on_workspace = str(resolved).startswith("/workspace")
    results_dir_env = os.environ.get("RESULTS_DIR", "")
    env_note = ""
    if results_dir_env and not results_dir_env.startswith("/workspace"):
        env_note = f" (note: RESULTS_DIR={results_dir_env} also does not point under /workspace)"
    return _check(
        "repo_on_network_volume",
        on_workspace,
        f"repo root resolves to {resolved}"
        + (
            ""
            if on_workspace
            else " -- not under /workspace, results will NOT survive a pod restart"
        )
        + env_note,
    )


def check_history_guard(config: dict[str, Any]) -> dict[str, Any]:
    from src.utils.history_guard import history_truncation_active
    from src.utils.step_config import resolve_step_fn_kwargs

    active_domains = []
    for domain in DOMAINS:
        step_cfg = resolve_step_fn_kwargs(config, domain)
        if history_truncation_active(step_cfg):
            active_domains.append(domain)
    ok = not active_domains
    return _check(
        "history_guard",
        ok,
        "no truncation params active in either domain's step config"
        if ok
        else f"truncation params active for: {active_domains} -- not valid for confirmatory H3 "
        "without --allow-history-truncation",
    )


def check_sidecar_and_concurrency(config: dict[str, Any]) -> dict[str, Any]:
    from src.utils.logprob_sidecar import parse_full_instance_overrides, parse_logprob_sidecar_mode
    from src.utils.manifest import load_manifest

    logging_cfg = config.get("logging") or {}
    problems: list[str] = []

    try:
        mode = parse_logprob_sidecar_mode(logging_cfg)
    except ValueError as e:
        return _check("sidecar_and_concurrency", False, f"invalid logprob_sidecar_mode: {e}")
    if mode != "action_window":
        problems.append(f"logprob_sidecar_mode={mode!r}, expected 'action_window' for production")

    try:
        full_overrides = parse_full_instance_overrides(logging_cfg)
    except ValueError as e:
        return _check(
            "sidecar_and_concurrency", False, f"invalid logprob_sidecar_full_instances: {e}"
        )

    for domain in DOMAINS:
        full_ids = full_overrides.get(domain) or full_overrides.get("*") or set()
        manifest = load_manifest(domain, config, REPO_ROOT)
        holdout_ids = {iid for iid, entry in manifest.items() if entry.get("holdout")}
        collision = full_ids & holdout_ids
        if collision:
            problems.append(
                f"{domain}: full-sidecar instances {sorted(collision)} are holdout instances"
            )

    exec_cfg = config.get("execution") or {}
    n = exec_cfg.get("max_concurrent_episodes")
    if n != EXPECTED_MAX_CONCURRENT_EPISODES:
        problems.append(
            f"execution.max_concurrent_episodes={n}, expected {EXPECTED_MAX_CONCURRENT_EPISODES} "
            "(C-1 batch-invariant frozen value)"
        )

    ok = not problems
    return _check(
        "sidecar_and_concurrency",
        ok,
        "sidecar_mode=action_window, full-instances don't collide with holdout, N="
        f"{EXPECTED_MAX_CONCURRENT_EPISODES}"
        if ok
        else "; ".join(problems),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/experiment_core.yaml")
    parser.add_argument("--out", default=None, help="Optional path to write the JSON report to.")
    args = parser.parse_args()

    from scripts.difficulty_calibration.sweep_textworld_difficulty import _load_merged_config

    config_path = (
        REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    )
    config = _load_merged_config(config_path)

    print(f"Gate F run-hygiene preflight — config={config_path}\n")
    results = [
        check_model_on_ephemeral_disk(config),
        check_langfuse(config),
        check_repo_on_network_volume(),
        check_history_guard(config),
        check_sidecar_and_concurrency(config),
    ]

    all_ok = all(r["pass"] for r in results)
    print(
        f"\n{'ALL CHECKS PASSED' if all_ok else 'FAILED'} ({sum(r['pass'] for r in results)}/{len(results)})"
    )

    if args.out:
        out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"config_path": str(config_path), "checks": results, "pass": all_ok}, indent=2
            )
        )
        print(f"Wrote {out_path}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
