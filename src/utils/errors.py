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
