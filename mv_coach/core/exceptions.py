"""Custom exceptions for MV-Coach-Eval."""


class MVCoachError(Exception):
    """Base exception for MV-Coach-Eval."""

    pass


class ConfigurationError(MVCoachError):
    """Raised when configuration is invalid."""

    pass


class DataLoadError(MVCoachError):
    """Raised when data loading fails."""

    pass


class ModelError(MVCoachError):
    """Raised when model operations fail."""

    pass


class RegistryError(MVCoachError):
    """Raised when model registry operations fail."""

    pass


class EvaluationError(MVCoachError):
    """Raised when evaluation operations fail."""

    pass
