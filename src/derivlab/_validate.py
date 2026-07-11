"""Shared input validation for pricing functions."""

from __future__ import annotations

import numpy as np

__all__ = [
    "validate_kind",
    "validate_style",
    "validate_positive",
    "validate_non_negative",
    "validate_spot_strike",
    "validate_sigma",
    "validate_T",
    "as_float_array",
    "broadcast_float_arrays",
    "scalar_or_array",
]


def validate_kind(kind: str) -> str:
    k = kind.lower()
    if k not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return k


def validate_style(style: str) -> str:
    s = style.lower()
    if s not in ("european", "american"):
        raise ValueError(f"style must be 'european' or 'american', got {style!r}")
    return s


def validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def validate_spot_strike(S: float, K: float) -> None:
    validate_positive("S", S)
    validate_positive("K", K)


def validate_sigma(sigma: float) -> None:
    validate_positive("sigma", sigma)


def validate_T(T: float, *, allow_zero: bool = False) -> None:
    if allow_zero:
        validate_non_negative("T", T)
    elif T <= 0:
        raise ValueError(f"T must be > 0, got {T}")


def as_float_array(*values: object) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(v, dtype=float) for v in values)


def broadcast_float_arrays(*values: object) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.atleast_1d(np.asarray(v, dtype=float)) for v in values)
    return tuple(np.broadcast_arrays(*arrays))


def scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    """Return a Python float when `value` is 0-D, else the array."""
    if value.ndim == 0:
        return float(value)
    return value
