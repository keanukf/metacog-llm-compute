#!/usr/bin/env python3
"""
Empirical pre-check gate for C1 CoT→Verify handoff parsing quality.

Runs short C1 episodes on the target deployment and summarizes the structured parser outcomes
via fields emitted into C1 call_detail:
  - draft_status: parsed|unparsed
  - parse_method: post_think|legacy_action_prefix|first_line_fallback|none

Usage (example):
  python scripts/run_c1_handoff_gate.py --config configs/pilot.yaml --pilot-mode cuda --real --output-dir data/results

This script is designed to be executed on the RunPod target model; local runs in mock mode are supported
for plumbing checks but do not represent the real failure rates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.base_agent import run_episode
from src.agent.compute_stages import get_step_fn
from src.utils.dotenv_loader import load_dotenv_if_present
from src.utils.experiment_env import make_experiment_env, resolve_textworld_game_path
from src.utils.pilot_config import load_pilot_config_with_lmstudio_override
from src.utils.run_output_layout import make_run_subdirectory
from src.utils.step_config import resolve_step_fn_kwargs

_ = load_dotenv_if_present(REPO_ROOT)


@dataclass(frozen=True)
class GateCounts:
    clean: int = 0
    recoverable_fallback: int = 0
    unparsed: int = 0

    @property
    def n(self) -> int:
        return int(self.clean + self.recoverable_fallback + self.unparsed)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n) + (z**2) / (4.0 * n * n)) ** 0.5
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def _create_real_model(config: dict, pilot_mode: str) -> Any:
    """Create model wrapper; mirrors scripts/run_pilot.py behaviour (minimal subset)."""
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name")
    if not model_name:
        raise RuntimeError("model.name missing in config")
    dtype = model_cfg.get("dtype", "float16")
    from src.utils.model_wrapper import create_wrapper

    if pilot_mode == "hf":
        return create_wrapper(backend="hf", model_name=model_name, dtype=dtype, device="mps")
    if pilot_mode == "cuda":
        inf = config.get("inference", {}) or {}
        backend = str(inf.get("backend", "vllm") or "vllm").strip().lower()
        if backend not in {"vllm", "hf"}:
            backend = "vllm"
        max_model_len = inf.get("max_model_len") or inf.get("vllm_max_model_len") or 8192
        extra = {
            "max_model_len": int(max_model_len),
            "chat_template": bool(inf.get("chat_template", True)),
            "enable_thinking": bool(inf.get("enable_thinking", False)),
        }
        gmu = inf.get("gpu_memory_utilization")
        if gmu is not None:
            try:
                extra["gpu_memory_utilization"] = float(gmu)
            except Exception:
                pass
        return create_wrapper(backend=backend, model_name=model_name, dtype=dtype, **extra)
    if pilot_mode == "lmstudio":
        inf = config.get("inference", {}) or {}
        base_url = inf.get("lmstudio_base_url") or os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        api_key = inf.get("lmstudio_api_key") or os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
        top_k = int(inf.get("lmstudio_top_logprobs", 5))
        return create_wrapper(
            backend="lmstudio",
            model_name=model_name,
            base_url=base_url,
            lmstudio_api_key=api_key,
            lmstudio_top_logprobs=top_k,
        )
    raise RuntimeError(f"Unsupported pilot_mode={pilot_mode!r}")


def _classify(parse_method: str, status: str) -> str:
    pm = (parse_method or "none").strip().lower()
    st = (status or "unparsed").strip().lower()
    if st != "parsed":
        return "unparsed"
    if pm == "post_think":
        return "clean"
    if pm in {"legacy_action_prefix", "first_line_fallback"}:
        return "recoverable_fallback"
    # Conservative fallback: treat any unknown parsed method as recoverable.
    return "recoverable_fallback"


def _count_from_trace(trace_path: Path) -> GateCounts:
    clean = fallback = unparsed = 0
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            cd = rec.get("call_detail")
            if not isinstance(cd, dict):
                continue
            if str(cd.get("stage") or "").strip().upper() != "C1":
                continue
            label = _classify(str(cd.get("parse_method") or ""), str(cd.get("draft_status") or ""))
            if label == "clean":
                clean += 1
            elif label == "recoverable_fallback":
                fallback += 1
            else:
                unparsed += 1
    return GateCounts(clean=clean, recoverable_fallback=fallback, unparsed=unparsed)


def _render_report(domain: str, counts: GateCounts, *, out_path: Path) -> None:
    n = counts.n
    unp = counts.unparsed
    unp_rate = (unp / n) if n else 0.0
    lo, hi = wilson_ci(unp, n)
    md = (
        f"# C1 handoff gate — {domain}\n\n"
        f"- N (steps counted): **{n}**\n"
        f"- clean (post_think): **{counts.clean}**\n"
        f"- recoverable_fallback (legacy_action_prefix|first_line_fallback): **{counts.recoverable_fallback}**\n"
        f"- unparsed: **{counts.unparsed}**\n\n"
        "## Unparsed rate\n\n"
        f"- unparsed_rate: **{unp_rate:.3%}**\n"
        f"- 95% Wilson CI: **[{lo:.3%}, {hi:.3%}]**\n"
    )
    out_path.write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run C1 handoff empirical gate and write markdown summary.")
    ap.add_argument("--config", default="configs/pilot.yaml", help="YAML config path")
    ap.add_argument("--output-dir", default="data/results", help="Base output directory")
    ap.add_argument("--no-timestamp-run", action="store_true", help="Write directly under --output-dir")
    ap.add_argument("--pilot-mode", default="mock", help="mock|hf|cuda|lmstudio")
    ap.add_argument("--real", action="store_true", help="Require real model (fail if not)")
    ap.add_argument("--domain", choices=["textworld", "tower_of_hanoi"], default="textworld")
    ap.add_argument("--n-episodes", type=int, default=10, help="Episodes to run (gate counts steps)")
    ap.add_argument("--max-steps", type=int, default=10, help="Max env steps per episode")
    args = ap.parse_args()

    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    base_output = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    output_dir = base_output if args.no_timestamp_run else make_run_subdirectory(base_output, prefix="c1_handoff_gate")

    config, _, _ = load_pilot_config_with_lmstudio_override(config_path, args.pilot_mode, REPO_ROOT, None)
    stage = "C1"
    step_cfg = resolve_step_fn_kwargs(config, args.domain)
    hist_keys = {
        "history_keep_last_pairs",
        "history_max_obs_chars",
        "history_current_obs_max_chars",
        "history_obs_head_ratio",
        "pin_recipe",
    }
    history_cfg = {k: step_cfg.pop(k) for k in list(step_cfg.keys()) if k in hist_keys}
    step_fn = get_step_fn(stage, save_logprob_distributions=False, save_vc_distributions=False, **step_cfg)

    if args.pilot_mode == "mock":
        model: Any = None
        if args.real:
            raise RuntimeError("--real requested but pilot_mode=mock")

        class MockModel:
            def generate(self, prompt: str, logprobs: bool = False, **kwargs):
                # Emit a mix of formats so the plumbing works; not a real gate.
                if "Before answering" in (prompt or ""):
                    text = "<think>ok</think>\ngo north"
                else:
                    text = "go north"
                lp = [{"logprob": -0.5}] * 4 if logprobs else None
                return text, lp

        model = MockModel()
    else:
        model = _create_real_model(config, args.pilot_mode)

    # Run episodes and write step traces to JSONL; then parse for gate statistics.
    trace_dir = output_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    domain = args.domain
    trace_path = trace_dir / f"c1_handoff_gate_{domain}_steps.jsonl"

    # Remove any old file in this run dir (safety).
    if trace_path.exists():
        trace_path.unlink()

    # Run N episodes; the gate counts steps from the trace file.
    for ep_i in range(int(args.n_episodes)):
        inst = ep_i
        if domain == "textworld":
            _ = resolve_textworld_game_path(inst, config, REPO_ROOT)
            env = make_experiment_env("textworld", inst, config, int(args.max_steps), REPO_ROOT)
        else:
            env = make_experiment_env("tower_of_hanoi", inst, config, int(args.max_steps), REPO_ROOT)
        ep_id = f"c1_handoff_gate_{domain}_{ep_i}"
        run_episode(
            env,
            model,
            stage,
            step_fn=step_fn,
            max_steps=int(args.max_steps),
            save_logprob_distributions=False,
            save_vc_distributions=False,
            save_step_traces=True,
            episode_id=ep_id,
            trace_output_dir=str(trace_dir),
            trace_model_name=str(config.get("model", {}).get("name", "")) or None,
            trace_hook=None,
            trace_session_id=str(output_dir.name),
            trace_tags=["c1_handoff_gate", domain, stage],
            trace_name=ep_id,
            **history_cfg,
        )
        # Merge per-episode trace lines into one file for easy counting.
        ep_trace = trace_dir / f"{ep_id}.jsonl"
        if ep_trace.is_file():
            with open(trace_path, "a", encoding="utf-8") as out_f:
                with open(ep_trace, "r", encoding="utf-8") as in_f:
                    for ln in in_f:
                        out_f.write(ln)

    counts = _count_from_trace(trace_path)
    out_md = output_dir / f"c1_handoff_gate_{domain}.md"
    _render_report(domain, counts, out_path=out_md)
    print(f"Wrote {out_md}")
    print(f"Wrote {trace_path}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    dt = time.perf_counter() - t0
    print(f"Done in {dt:.2f}s")

