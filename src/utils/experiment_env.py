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
    backend = config.get("inference", {}).get("backend", "vllm")
    try:
        from src.utils.model_wrapper import create_wrapper

        return create_wrapper(backend=backend, model_name=model_name, dtype=dtype)
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

        cfg = config.get("tower_of_hanoi", {})
        num_disks_range = cfg.get("num_disks_range", [3, 4])
        partial_start_range = cfg.get("partial_start_range", [0, 3])
        base_seed = int(cfg.get("task_generation_seed", 42))
        seed = base_seed + instance * 10007
        task_instance = generate_instances(
            1,
            seed=seed,
            num_disks_range=(int(num_disks_range[0]), int(num_disks_range[1])),
            partial_start_range=(int(partial_start_range[0]), int(partial_start_range[1])),
        )[0]
        include_vm = bool(dom_cfg.get("include_valid_moves", False)) if dom_cfg else False
        return TowerOfHanoiEnv(task=task_instance, max_steps=max_steps, include_valid_moves=include_vm)
    return TextWorldEnv(game_file=None, max_steps=max_steps)
