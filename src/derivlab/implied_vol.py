"""Implied volatility inversion.

Newton-Raphson on vega with a Brent bracketing fallback. Prices that
violate static no-arbitrage bounds return NaN rather than a garbage vol —
essential when feeding real market quotes, which routinely contain stale
or crossed prices at the wings.

Accepts scalars or NumPy arrays (broadcasting applies).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from . import bs
from ._validate import broadcast_float_arrays, scalar_or_array, validate_kind

__all__ = ["implied_vol"]

_VOL_LO, _VOL_HI = 1e-4, 5.0


def _no_arb_bounds(
    S: NDArray[np.floating[Any]],
    K: NDArray[np.floating[Any]],
    T: NDArray[np.floating[Any]],
    r: float,
    q: float,
    kind: str,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    if kind == "call":
        lo = np.maximum(S * df_q - K * df_r, 0.0)
        hi = S * df_q
    else:
        lo = np.maximum(K * df_r - S * df_q, 0.0)
        hi = K * df_r
    return lo, hi


def _solve_scalar(
    target: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kind: str,
    tol: float,
    max_iter: int,
) -> float:
    lo, hi = _no_arb_bounds(
        np.array([S]), np.array([K]), np.array([T]), r, q, kind
    )
    if not (lo[0] < target < hi[0]):
        return float("nan")

    sigma = float(np.clip(np.sqrt(2.0 * np.pi / T) * target / S, 0.05, 2.0))
    for _ in range(max_iter):
        px = bs.price(S, K, T, r, sigma, q, kind)
        vega = bs.greeks(S, K, T, r, sigma, q, kind)["vega"]
        diff = float(px) - target
        if abs(diff) < tol:
            return sigma
        if float(vega) < 1e-12:
            break
        sigma -= diff / float(vega)
        if not (_VOL_LO < sigma < _VOL_HI):
            break

    def f(s: float) -> float:
        return float(bs.price(S, K, T, r, s, q, kind)) - target

    f_lo, f_hi = f(_VOL_LO), f(_VOL_HI)
    if f_lo * f_hi > 0:
        return float("nan")
    return float(brentq(f, _VOL_LO, _VOL_HI, xtol=1e-12))


def _solve_batch(
    target: NDArray[np.floating[Any]],
    S: NDArray[np.floating[Any]],
    K: NDArray[np.floating[Any]],
    T: NDArray[np.floating[Any]],
    r: float,
    q: float,
    kind: str,
    tol: float,
    max_iter: int,
) -> NDArray[np.floating[Any]]:
    out = np.full(target.shape, np.nan, dtype=float)
    lo, hi = _no_arb_bounds(S, K, T, r, q, kind)
    valid = (T > 0) & (target > lo) & (target < hi)
    for i in np.flatnonzero(valid):
        out.flat[i] = _solve_scalar(
            float(target.flat[i]),
            float(S.flat[i]),
            float(K.flat[i]),
            float(T.flat[i]),
            r,
            q,
            kind,
            tol,
            max_iter,
        )
    return out


def implied_vol(
    target_price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    q: float = 0.0,
    kind: str = "call",
    tol: float = 1e-10,
    max_iter: int = 60,
) -> float | NDArray[np.floating[Any]]:
    """Solve for the BSM volatility that reproduces `target_price`.

    Returns NaN for prices outside no-arbitrage bounds or T <= 0.
    """
    kind = validate_kind(kind)
    target, S_arr, K_arr, T_arr = broadcast_float_arrays(target_price, S, K, T)
    if target.size == 1:
        t_val = float(T_arr.flat[0])
        if t_val <= 0:
            return float("nan")
        return _solve_scalar(
            float(target.flat[0]),
            float(S_arr.flat[0]),
            float(K_arr.flat[0]),
            t_val,
            r,
            q,
            kind,
            tol,
            max_iter,
        )

    out = np.full(target.shape, np.nan, dtype=float)
    pos = T_arr > 0
    if not np.any(pos):
        return scalar_or_array(out)
    solved = _solve_batch(
        target[pos], S_arr[pos], K_arr[pos], T_arr[pos], r, q, kind, tol, max_iter
    )
    out[pos] = solved
    return scalar_or_array(out)
