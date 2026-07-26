from typing import Any


def safe_get(data: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path out of nested dicts, returning `default` on any miss.

    CTGov study JSON omits whole modules/fields inconsistently, so every read
    in this codebase goes through here rather than chained `.get()` calls.
    """
    if not path:
        return default

    current = data
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return default
        current = current[segment]
    return current
