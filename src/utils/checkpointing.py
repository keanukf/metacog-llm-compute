"""
Episode-level checkpointing. On pod crash, at most one episode is lost.
Scripts check list_completed_episodes() and resume from remaining work.
"""
from __future__ import annotations

from pathlib import Path


def list_completed_episodes(checkpoint_dir: str | Path) -> set[str]:
    """
    List episode IDs that already have a checkpoint JSON in the given directory.

    Args:
        checkpoint_dir: Directory containing episode JSON files (e.g. ep_*.json).

    Returns:
        Set of episode IDs (filenames without .json).
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return set()
    return {p.stem for p in checkpoint_dir.glob("*.json")}


def save_episode_checkpoint(
    checkpoint_dir: str | Path,
    episode_id: str,
    data: dict,
) -> Path:
    """
    Save one episode's result as a checkpoint file.
    Typically called via logging_utils.log_episode() with the same data.

    Args:
        checkpoint_dir: Results directory.
        episode_id: Unique episode id.
        data: Episode result dict to persist.

    Returns:
        Path to the written JSON file.
    """
    from src.utils.logging_utils import log_episode

    return log_episode(episode_id, data, checkpoint_dir)
