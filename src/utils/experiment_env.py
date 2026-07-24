"""
Shared environment and model construction for phase runners (Phase 1 / Phase 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.compute_stages import VC_FOLLOWUP_PROMPT_MARKER


def resolve_textworld_game_path(instance: int, config: dict, repo_root: Path) -> Path | None:
    """
    Locate a compiled TextWorld story for ``instance`` (0-based index).

    Searches ``paths.tasks_dir`` (default ``data/tasks``) for, in order:
    ``textworld_{instance}.z8``, ``textworld/textworld_{instance}.z8``, same for ``.ulx``.
    Paths are resolved relative to ``repo_root`` when not absolute.
    """
    tasks_dir = Path(config.get("paths", {}).get("tasks_dir", "data/tasks"))
    candidate_paths = [
        tasks_dir / f"textworld_{instance}.z8",
        tasks_dir / "textworld" / f"textworld_{instance}.z8",
        tasks_dir / f"textworld_{instance}.ulx",
        tasks_dir / "textworld" / f"textworld_{instance}.ulx",
    ]
    for cand in candidate_paths:
        cand_abs = cand if cand.is_absolute() else repo_root / cand
        if cand_abs.is_file():
            return cand_abs
    return None


def assert_textworld_games_present(instances: int, config: dict, repo_root: Path) -> None:
    """
    Fail loudly before a real run starts if any TextWorld instance in ``range(instances)``
    has no resolvable compiled game file.

    Without this, ``TextWorldEnv`` silently falls back to an unwinnable stub environment per
    missing instance (see ``_use_real`` in ``src/environments/textworld_env.py``) instead of
    erroring — which let 45/50 frozen instances run as unwinnable stubs, undetected, for the
    full 2026-07-22 Phase 1 collection (see docs/consistency_log.md).
    """
    missing = [i for i in range(instances) if resolve_textworld_game_path(i, config, repo_root) is None]
    if missing:
        preview = missing[:10]
        more = "..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"TextWorld game files missing for {len(missing)}/{instances} instances "
            f"(instance ids: {preview}{more}). Generate the frozen manifest set first, e.g.:\n"
            "  python scripts/datasets/generate_textworld_games.py --num-rooms 5 --num-ingredients 1 "
            "--cook --seed 20260718 --num-instances 50 --output-dir data/tasks/textworld\n"
            "(the exact frozen seed/params are in docs/consistency_log.md's TextWorld freeze entry — "
            "do not use the docs/runpod.md Step 6b example command as-is, it generates only 5 "
            "smoke-test instances with a different seed)."
        )


class MockExperimentModel:
    """Lightweight stand-in when ``--real`` is off or model creation fails."""

    def generate(self, prompt, logprobs=False, **kwargs):
        # VC follow-up prompt (see ``compute_stages.VC_FOLLOWUP_PROMPT_MARKER``)
        if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
            text = "75"
            lp = (
                [{"token": "7", "logprob": -0.1}, {"token": "5", "logprob": -0.05}]
                if logprobs
                else None
            )
            return text, lp
        text = "go north"
        lp = [{"logprob": -0.5}] * 5 if logprobs else None
        return text, lp


def create_experiment_model(config: dict, use_real: bool) -> Any:
    """Create model wrapper from config; real if ``use_real`` and setup succeeds, else mock."""
    if not use_real:
        return MockExperimentModel()
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name")
    if not model_name:
        return MockExperimentModel()
    dtype = model_cfg.get("dtype", "float16")
    if str(dtype).lower() == "fp16":
        dtype = "float16"
    revision = model_cfg.get("revision")
    backend = config.get("inference", {}).get("backend", "vllm")
    try:
        from src.utils.model_wrapper import create_wrapper

        extra: dict[str, Any] = {}
        if revision:
            extra["revision"] = str(revision)
        inf = config.get("inference", {}) or {}
        if backend == "vllm":
            extra["chat_template"] = bool(inf.get("chat_template", True))
            extra["enable_thinking"] = bool(inf.get("enable_thinking", False))
            # Match run_pilot.py: models may advertise context lengths that do not fit
            # KV cache on 24 GB GPUs (e.g. Qwen3-8B @ 40960).
            max_model_len = inf.get("max_model_len") or inf.get("vllm_max_model_len")
            if max_model_len is None:
                max_model_len = 8192
            try:
                extra["max_model_len"] = int(max_model_len)
            except (TypeError, ValueError):
                extra["max_model_len"] = 8192
            gmu = inf.get("gpu_memory_utilization")
            if gmu is not None:
                try:
                    extra["gpu_memory_utilization"] = float(gmu)
                except (TypeError, ValueError):
                    pass
        from src.utils.inference.logprob_config import resolve_top_logprobs

        extra["top_logprobs"] = resolve_top_logprobs(inf)
        return create_wrapper(backend=backend, model_name=model_name, dtype=dtype, **extra)
    except Exception:
        return MockExperimentModel()


def make_experiment_env(
    domain: str,
    instance: int,
    config: dict,
    max_steps: int,
    repo_root: Path,
) -> Any:
    """Create environment for calibration / phase-2 runs (TextWorld, Tower of Hanoi)."""
    from src.environments.textworld_env import TextWorldEnv

    dom_cfg: dict[str, Any] | None = None
    domain_prompts = config.get("domain_prompts")
    if isinstance(domain_prompts, dict):
        raw = domain_prompts.get(domain)
        if isinstance(raw, dict):
            dom_cfg = raw

    if domain == "textworld":
        game_file = resolve_textworld_game_path(instance, config, repo_root)
        include_adm = bool(dom_cfg.get("include_admissible_commands", False)) if dom_cfg else False
        return TextWorldEnv(
            game_file=str(game_file) if game_file else None,
            max_steps=max_steps,
            include_admissible_commands=include_adm,
        )
    if domain == "tower_of_hanoi":
        from src.environments.tower_of_hanoi import TowerOfHanoiEnv, generate_instances
        from src.utils.manifest import load_manifest, manifest_entry_for_instance

        cfg = config.get("tower_of_hanoi", {})
        manifest = manifest_entry_for_instance(domain, instance, config, repo_root)
        if manifest or load_manifest(domain, config, repo_root):
            base_seed = int(
                manifest.get("task_generation_seed") or cfg.get("task_generation_seed", 42)
            )
            # Prefer the manifest's own generation parameters over config: these are what the
            # frozen manifest was actually built with, and must match exactly or
            # generate_instances() silently reconstructs a *different* instance at this index
            # (e.g. defaulting back to partial_start_mode="optimal_prefix" would collapse the
            # random_scramble state space back down to the num_disks+1-state ceiling).
            num_disks_range = manifest.get("num_disks_range") or cfg.get("num_disks_range", [3, 4])
            partial_start_range = manifest.get("partial_start_range") or cfg.get(
                "partial_start_range", [0, 3]
            )
            partial_start_mode = manifest.get("partial_start_mode") or cfg.get(
                "partial_start_mode", "optimal_prefix"
            )
            all_manifest = load_manifest(domain, config, repo_root)
            n_inst = len(all_manifest) if all_manifest else max(50, instance + 1)
            task_instance = generate_instances(
                n_inst,
                seed=base_seed,
                num_disks_range=(int(num_disks_range[0]), int(num_disks_range[1])),
                partial_start_range=(int(partial_start_range[0]), int(partial_start_range[1])),
                partial_start_mode=str(partial_start_mode),
            )[instance]
        else:
            import warnings

            warnings.warn(
                f"No ToH manifest for instance {instance}; using legacy per-instance seed.",
                UserWarning,
                stacklevel=2,
            )
            num_disks_range = cfg.get("num_disks_range", [3, 4])
            partial_start_range = cfg.get("partial_start_range", [0, 3])
            base_seed = int(cfg.get("task_generation_seed", 42))
            seed = base_seed + instance * 10007
            task_instance = generate_instances(
                1,
                seed=seed,
                num_disks_range=(int(num_disks_range[0]), int(num_disks_range[1])),
                partial_start_range=(int(partial_start_range[0]), int(partial_start_range[1])),
                partial_start_mode=str(cfg.get("partial_start_mode", "optimal_prefix")),
            )[0]
        include_vm = bool(dom_cfg.get("include_valid_moves", False)) if dom_cfg else False
        # Prefer the per-instance cap generate_instances() computes (3x optimal_steps) over the
        # flat config/CLI value -- this is what the Gate D corridor calibration actually tested
        # and froze (docs/consistency_log.md, 2026-07-20 Gate F budget re-estimate finding); a flat
        # cap would silently diverge from the validated success rates.
        instance_max_steps = task_instance.get("max_steps")
        effective_max_steps = int(instance_max_steps) if instance_max_steps else max_steps
        return TowerOfHanoiEnv(
            task=task_instance, max_steps=effective_max_steps, include_valid_moves=include_vm
        )
    return TextWorldEnv(game_file=None, max_steps=max_steps)
