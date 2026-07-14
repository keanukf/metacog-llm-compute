"""Logprob sidecar mode resolution and action-window filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.signals.token_entropy import slice_action_logprob_tokens

LogprobSidecarMode = Literal["off", "action_window", "full"]

_VALID_MODES = frozenset({"off", "action_window", "full"})

LEGACY_SAVE_LOGPROB_DISTRIBUTIONS_ERROR = (
    "logging.save_logprob_distributions is deprecated; set logging.logprob_sidecar_mode "
    "to one of off, action_window, full instead. Silent fallback to full sidecars "
    "(~522 GB at Phase 1/2 scale) is not allowed."
)


def _normalize_mode(raw: str | None) -> LogprobSidecarMode | None:
    if raw is None:
        return None
    mode = str(raw).strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"logprob_sidecar_mode must be one of {sorted(_VALID_MODES)}, got {raw!r}")
    return mode  # type: ignore[return-value]


def parse_logprob_sidecar_mode(logging: dict[str, Any] | None) -> LogprobSidecarMode:
    """Resolve default sidecar mode from ``logging.logprob_sidecar_mode`` (default ``off``)."""
    lg = logging or {}
    if "save_logprob_distributions" in lg:
        raise ValueError(LEGACY_SAVE_LOGPROB_DISTRIBUTIONS_ERROR)
    explicit = _normalize_mode(lg.get("logprob_sidecar_mode"))
    if explicit is not None:
        return explicit
    return "off"


def parse_full_instance_overrides(logging: dict[str, Any] | None) -> dict[str, set[int]]:
    """
    Instances that receive ``full`` sidecars (reasoning retained) per domain.

    Accepts ``logprob_sidecar_full_instances`` as:
    - ``{"textworld": [0], "tower_of_hanoi": [0]}``
    - ``[0, 1]`` (applied to all domains)
    """
    lg = logging or {}
    raw = lg.get("logprob_sidecar_full_instances")
    if raw is None:
        return {}
    if isinstance(raw, list):
        ids = {int(x) for x in raw}
        return {"*": ids}
    if isinstance(raw, dict):
        out: dict[str, set[int]] = {}
        for dom, vals in raw.items():
            if isinstance(vals, list):
                out[str(dom)] = {int(x) for x in vals}
        return out
    raise ValueError("logprob_sidecar_full_instances must be a list or domain→list map")


@dataclass(frozen=True)
class LogprobSidecarConfig:
    """Per-run logprob sidecar policy (default mode + full-reasoning overrides)."""

    default_mode: LogprobSidecarMode
    full_instances_by_domain: dict[str, set[int]]
    export_format: str = "json"
    subdir: str = "logprobs"

    @classmethod
    def from_logging_config(cls, logging: dict[str, Any] | None) -> LogprobSidecarConfig:
        lg = logging or {}
        return cls(
            default_mode=parse_logprob_sidecar_mode(lg),
            full_instances_by_domain=parse_full_instance_overrides(lg),
            export_format=str(lg.get("logprob_export_format", "json")).lower(),
            subdir=str(lg.get("logprob_subdir", "logprobs")),
        )

    def mode_for(self, domain: str, instance: int) -> LogprobSidecarMode:
        inst = int(instance)
        dom = str(domain)
        wildcard = self.full_instances_by_domain.get("*", set())
        domain_set = self.full_instances_by_domain.get(dom, set())
        if inst in wildcard or inst in domain_set:
            return "full"
        return self.default_mode

    def capture_enabled(self, domain: str, instance: int) -> bool:
        return self.mode_for(domain, instance) != "off"


def filter_logprob_raw_for_sidecar(
    logprob_raw_per_step: list[Any] | None,
    mode: LogprobSidecarMode,
) -> list[Any]:
    """Apply sidecar scope before persistence (``full`` passes through unchanged)."""
    if not logprob_raw_per_step or mode == "off":
        return []
    if mode == "full":
        return list(logprob_raw_per_step)

    filtered: list[Any] = []
    for lp in logprob_raw_per_step:
        if isinstance(lp, list) and (not lp or isinstance(lp[0], dict)):
            filtered.append(slice_action_logprob_tokens(lp))
            continue
        if isinstance(lp, list) and lp and (lp[0] is None or isinstance(lp[0], list)):
            samples = [
                slice_action_logprob_tokens(s_lp) if isinstance(s_lp, list) else [] for s_lp in lp
            ]
            filtered.append(samples)
            continue
        filtered.append(lp)
    return filtered
