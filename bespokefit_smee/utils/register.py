"""Utilities for registering functions."""

from typing import Any, Callable

from .typing import FnTypeVar


def get_registry_decorator(
    registry: dict[Any, Any],
) -> Callable[[Any], Callable[[FnTypeVar], FnTypeVar]]:
    """Get a decorator to register functions in a given registry.

    Parameters
    ----------
    registry : dict
        The registry to register functions in.

    Returns
    -------
    Callable[[Any], Callable[[FnTypeVar], FnTypeVar]]
        A decorator to register functions in the registry.
    """

    def register_fn(
        key: Any,
    ) -> Callable[[FnTypeVar], FnTypeVar]:
        """Decorator to register a function in a given registry.

        Parameters
        ----------
        key : Any
            The key to register the function under.

        Returns
        -------
        Callable[[FnTypeVar], FnTypeVar]
            A decorator that registers the function in the registry.
        """

        def decorator(func: FnTypeVar) -> FnTypeVar:
            if key in registry:
                raise ValueError(f"A function is already registered for {key}.")
            registry[key] = func
            return func

        return decorator

    return register_fn
