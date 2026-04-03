"""
Shared environment and model construction for phase runners (Phase 1 / Phase 2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class MockExperimentModel:
    """Lightweight stand-in when ``--real`` is off or model creation fails."""

    def generate(self, prompt, logprobs=False, **kwargs):
        # VC follow-up prompt (matches compute_stages._vc_followup_prompt)
        if "You just chose the action:" in (prompt or ""):
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

    if domain == "textworld":
        tasks_dir = Path(config.get("paths", {}).get("tasks_dir", "data/tasks"))
        candidate_paths = [
            tasks_dir / f"textworld_{instance}.z8",
            tasks_dir / "textworld" / f"textworld_{instance}.z8",
            tasks_dir / f"textworld_{instance}.ulx",
            tasks_dir / "textworld" / f"textworld_{instance}.ulx",
        ]
        game_file = None
        for cand in candidate_paths:
            cand_abs = cand if cand.is_absolute() else repo_root / cand
            if cand_abs.exists():
                game_file = cand_abs
                break
        return TextWorldEnv(game_file=str(game_file) if game_file else None, max_steps=max_steps)
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
        return TowerOfHanoiEnv(task=task_instance, max_steps=max_steps)
    return TextWorldEnv(game_file=None, max_steps=max_steps)
