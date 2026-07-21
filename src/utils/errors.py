"""Repository-specific exception hierarchy.

All rooted at ``MetacogError`` so callers (episode scheduler, backends, env labeling) can catch the
whole family, and ``run_resilience.classify_exclusion_reason`` can distinguish an infrastructure
fault worth quarantining from a data-labeling failure.
"""

from __future__ import annotations


class MetacogError(Exception):
    """Base exception for repository-specific runtime failures."""


class ConfigError(MetacogError):
    """Raised when configuration values are invalid or missing."""


class BackendError(MetacogError):
    """Raised when model/backend initialization or inference fails."""


class EnvironmentError(MetacogError):
    """Raised when environment setup/interaction fails."""


class ArtifactError(MetacogError):
    """Raised when reading/writing run artifacts fails."""


class EnvStateError(MetacogError):
    """Raised when environment init or step() invariants are violated."""


class LabelError(MetacogError):
    """Raised when BFS labeling cannot produce a valid optimal path."""
