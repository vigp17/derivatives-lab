"""Cox-Ross-Rubinstein binomial tree pricing.

Handles European and American exercise. Vectorized backward induction:
the option value vector at step i has i+1 nodes; each sweep is a single
NumPy expression, so even 5,000-step trees price in milliseconds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ._validate import (
    validate_kind,
    validate_sigma,
    validate_spot_strike,
    validate_style,
)

__all__ = ["price"]


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
    style: str = "european",
    steps: int = 800,
) -> float:
    kind = validate_kind(kind)
    style = validate_style(style)
    validate_spot_strike(S, K)
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if T <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    validate_sigma(sigma)

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(
            "risk-neutral probability outside (0,1); increase steps or check parameters"
        )

    if kind == "call":

        def payoff(s: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
            return cast(NDArray[np.floating[Any]], np.maximum(s - K, 0.0))

    else:

        def payoff(s: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
            return cast(NDArray[np.floating[Any]], np.maximum(K - s, 0.0))

    payoff_fn: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]] = payoff

    j = np.arange(steps + 1)
    stock = S * u**j * d ** (steps - j)
    value = payoff_fn(stock)

    for i in range(steps - 1, -1, -1):
        value = disc * (p * value[1:] + (1.0 - p) * value[:-1])
        if style == "american":
            j = np.arange(i + 1)
            stock = S * u**j * d ** (i - j)
            value = np.maximum(value, payoff_fn(stock))

    return float(value[0])
