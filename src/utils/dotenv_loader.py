"""
Load environment variables from a local ``.env`` file (optional).

This is meant for developer convenience when running locally (LM Studio, Langfuse),
and is safe on servers: if ``python-dotenv`` is not installed or ``.env`` is missing,
this becomes a no-op.

By default ``override=True`` so values from ``.env`` replace variables already present
in the process environment. That avoids a common failure mode: IDEs or shells pre-defining
empty ``LANGFUSE_*`` (or other) keys, which makes ``load_dotenv(override=False)`` skip those
lines, return False, and leave secrets unset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_dotenv_if_present(repo_root: str | Path, *, override: bool = True) -> dict[str, Any]:
    """
    Attempt to load ``.env`` from ``repo_root`` using python-dotenv.

    Args:
        repo_root: Repository root path (CWD for scripts).
        override: If True (default), .env values override existing environment variables.

    Returns:
        Dict with keys:
        - loaded: bool (True if python-dotenv applied at least one variable from the file)
        - reason: str (when not loaded / nothing applied)
        - env_path: str (candidate path)
    """
    root = Path(repo_root)
    env_path = root / ".env"
    if not env_path.exists():
        return {"loaded": False, "reason": "no .env file", "env_path": str(env_path)}
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception as e:  # pragma: no cover
        return {
            "loaded": False,
            "reason": f"python-dotenv unavailable: {e!s}",
            "env_path": str(env_path),
        }

    # utf-8-sig strips a leading UTF-8 BOM so the first key is not "\ufeffLANGFUSE_..." (which breaks os.environ lookups).
    ok = bool(
        load_dotenv(
            dotenv_path=env_path,
            override=bool(override),
            encoding="utf-8-sig",
        )
    )
    if ok:
        return {"loaded": True, "reason": "", "env_path": str(env_path)}
    # File exists but python-dotenv parsed no assignments (e.g. empty file, comments only).
    return {
        "loaded": False,
        "reason": "no variables parsed from .env (empty, comment-only, or invalid syntax)",
        "env_path": str(env_path),
    }
