"""Black-Scholes-Merton analytic pricing and Greeks.

All functions accept scalars or NumPy arrays (broadcasting applies).
Conventions:
    S     spot price
    K     strike
    T     time to expiry in years (must be > 0 for Greeks)
    r     continuously compounded risk-free rate
    sigma annualized volatility
    q     continuous dividend yield
Vega and rho are per unit change (multiply by 0.01 for per-1%).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from ._validate import (
    as_float_array,
    scalar_or_array,
    validate_kind,
    validate_sigma,
    validate_spot_strike,
)

__all__ = ["price", "greeks", "d1_d2"]


def d1_d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """The d1/d2 terms of the BSM formula."""
    S_arr, K_arr, T_arr, sigma_arr = as_float_array(S, K, T, sigma)
    if np.any(S_arr <= 0) or np.any(K_arr <= 0):
        raise ValueError("S and K must be > 0")
    if np.any(sigma_arr <= 0):
        raise ValueError("sigma must be > 0")
    sqrt_t = np.sqrt(T_arr)
    d1 = (np.log(S_arr / K_arr) + (r - q + 0.5 * sigma_arr**2) * T_arr) / (
        sigma_arr * sqrt_t
    )
    return d1, d1 - sigma_arr * sqrt_t


def price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    kind: str = "call",
) -> float | NDArray[np.floating[Any]]:
    """European option price under BSM.

    T <= 0 returns intrinsic value (expiry).
    """
    kind = validate_kind(kind)
    S_arr, K_arr, T_arr, sigma_arr = as_float_array(S, K, T, sigma)
    intrinsic = (
        np.maximum(S_arr - K_arr, 0.0)
        if kind == "call"
        else np.maximum(K_arr - S_arr, 0.0)
    )
    if np.all(T_arr <= 0):
        return cast("float | NDArray[np.floating[Any]]", scalar_or_array(intrinsic))

    pos = (S_arr > 0) & (K_arr > 0) & (sigma_arr > 0)
    if np.any(~pos & (T_arr > 0)):
        raise ValueError("S, K, and sigma must be > 0 for T > 0")

    d1, d2 = d1_d2(S_arr, K_arr, np.where(T_arr > 0, T_arr, 1.0), r, sigma_arr, q)
    df_r = np.exp(-r * T_arr)
    df_q = np.exp(-q * T_arr)
    if kind == "call":
        val = S_arr * df_q * norm.cdf(d1) - K_arr * df_r * norm.cdf(d2)
    else:
        val = K_arr * df_r * norm.cdf(-d2) - S_arr * df_q * norm.cdf(-d1)
    out = np.where(T_arr > 0, val, intrinsic)
    return cast("float | NDArray[np.floating[Any]]", scalar_or_array(out))


def greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    kind: str = "call",
) -> dict[str, float | NDArray[np.floating[Any]]]:
    """Analytic Greeks: delta, gamma, vega, theta (per year), rho.

    Gamma and vega are identical for calls and puts.
    """
    kind = validate_kind(kind)
    S_arr, K_arr, T_arr, sigma_arr = as_float_array(S, K, T, sigma)
    if np.any(T_arr <= 0):
        raise ValueError("Greeks require T > 0")
    validate_spot_strike(float(np.min(S_arr)), float(np.min(K_arr)))
    validate_sigma(float(np.min(sigma_arr)))

    d1, d2 = d1_d2(S_arr, K_arr, T_arr, r, sigma_arr, q)
    sqrt_t = np.sqrt(T_arr)
    df_r = np.exp(-r * T_arr)
    df_q = np.exp(-q * T_arr)
    pdf_d1 = norm.pdf(d1)

    gamma = df_q * pdf_d1 / (S_arr * sigma_arr * sqrt_t)
    vega = S_arr * df_q * pdf_d1 * sqrt_t
    theta_common = -S_arr * df_q * pdf_d1 * sigma_arr / (2.0 * sqrt_t)

    if kind == "call":
        delta = df_q * norm.cdf(d1)
        theta = theta_common - r * K_arr * df_r * norm.cdf(d2) + q * S_arr * df_q * norm.cdf(
            d1
        )
        rho = K_arr * T_arr * df_r * norm.cdf(d2)
    else:
        delta = df_q * (norm.cdf(d1) - 1.0)
        theta = theta_common + r * K_arr * df_r * norm.cdf(-d2) - q * S_arr * df_q * norm.cdf(
            -d1
        )
        rho = -K_arr * T_arr * df_r * norm.cdf(-d2)

    return {
        "delta": scalar_or_array(delta),
        "gamma": scalar_or_array(gamma),
        "vega": scalar_or_array(vega),
        "theta": scalar_or_array(theta),
        "rho": scalar_or_array(rho),
    }
