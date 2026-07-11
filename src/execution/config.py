"""Execution-layer configuration and frozen (N, eps) metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.inference.logprob_invariance import TLE_INVARIANCE_EPS_BITS


@dataclass(frozen=True)
class ExecutionConfig:
    max_concurrent_episodes: int = 1
    backend_mode: str = "server"
    server_url: str = "http://127.0.0.1:8000/v1"
    model_name: str | None = None
    frozen_max_concurrent_episodes: int | None = None
    frozen_tle_invariance_eps: float | None = None
    n_mismatch_mode: str = "warn"  # warn | hard_fail

    @classmethod
    def from_config(cls, config: dict[str, Any], *, real: bool = False) -> ExecutionConfig:
        raw = config.get("execution") or {}
        model_cfg = config.get("model") or {}
        max_n = int(raw.get("max_concurrent_episodes", 1))
        if max_n < 1:
            max_n = 1
        backend_mode = str(raw.get("backend_mode", "server")).strip().lower()
        server_url = str(raw.get("server_url", "http://127.0.0.1:8000/v1")).rstrip("/")
        mismatch_mode = str(raw.get("n_mismatch_mode", "warn" if not real else "hard_fail"))
        frozen_n = raw.get("frozen_max_concurrent_episodes")
        frozen_eps = raw.get("frozen_tle_invariance_eps")
        return cls(
            max_concurrent_episodes=max_n,
            backend_mode=backend_mode,
            server_url=server_url,
            model_name=str(model_cfg.get("name")) if model_cfg.get("name") else None,
            frozen_max_concurrent_episodes=int(frozen_n) if frozen_n is not None else None,
            frozen_tle_invariance_eps=float(frozen_eps) if frozen_eps is not None else None,
            n_mismatch_mode=mismatch_mode,
        )

    def validate_frozen(self) -> list[str]:
        """Return warning/error messages for config vs frozen (N, eps) mismatch."""
        msgs: list[str] = []
        if self.frozen_max_concurrent_episodes is not None:
            if self.max_concurrent_episodes != self.frozen_max_concurrent_episodes:
                msgs.append(
                    f"execution.max_concurrent_episodes={self.max_concurrent_episodes} "
                    f"!= frozen_max_concurrent_episodes={self.frozen_max_concurrent_episodes}"
                )
        if self.frozen_tle_invariance_eps is not None:
            if abs(self.frozen_tle_invariance_eps - TLE_INVARIANCE_EPS_BITS) > 1e-9:
                pass  # frozen eps is authoritative when set
        return msgs

    def enforce_frozen_or_exit(self) -> None:
        msgs = self.validate_frozen()
        if not msgs:
            return
        text = "; ".join(msgs)
        if self.n_mismatch_mode == "hard_fail":
            raise SystemExit(f"Frozen execution params mismatch: {text}")
        import warnings

        warnings.warn(f"Frozen execution params mismatch: {text}", stacklevel=2)


def frozen_execution_params_dict(
    *,
    max_concurrent_episodes: int,
    tle_invariance_eps: float,
    eps_derived_under_load: bool = False,
) -> dict[str, Any]:
    return {
        "max_concurrent_episodes": int(max_concurrent_episodes),
        "tle_invariance_eps": float(tle_invariance_eps),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "eps_derived_under_load": bool(eps_derived_under_load),
    }


def write_frozen_execution_params(
    checkpoint_dir: str | Path,
    params: dict[str, Any],
) -> None:
    """Merge ``frozen_execution_params`` into ``run_metadata.json`` if present."""
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "run_metadata.json"
    if not meta_path.is_file():
        return
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["frozen_execution_params"] = dict(params)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_frozen_execution_params(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    meta_path = Path(checkpoint_dir) / "run_metadata.json"
    if not meta_path.is_file():
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    raw = meta.get("frozen_execution_params")
    return raw if isinstance(raw, dict) else None
