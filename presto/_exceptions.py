"""Exceptions for presto."""


class InvalidSettingsError(ValueError):
    """Exception raised for invalid settings."""


class MoleculeParameterisationError(RuntimeError):
    """Exception raised when one or more molecules cannot be parameterised."""
