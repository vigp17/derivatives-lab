"""Monte Carlo pricing under GBM with variance reduction.

Two techniques, composable:
  * Antithetic variates — each draw Z is paired with -Z; the estimator is
    the mean of pair-averages, which are i.i.d., so the standard error is
    computed over pairs (the statistically correct way).
  * Control variates — the discounted terminal stock price has known
    expectation S*exp(-qT); regressing the payoff against it removes the
    variance explained by the underlying's own randomness.

Every result carries its standard error. A Monte Carlo price without an
error bar is a random number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validate import validate_kind, validate_sigma, validate_spot_strike, validate_T

__all__ = ["MCResult", "price"]


@dataclass(frozen=True)
class MCResult:
    price: float
    std_err: float
    n_paths: int
    antithetic: bool
    control_variate: bool

    def ci(self, z: float = 1.96) -> tuple[float, float]:
        return self.price - z * self.std_err, self.price + z * self.std_err


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
    n_paths: int = 200_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> MCResult:
    kind = validate_kind(kind)
    validate_spot_strike(S, K)
    validate_T(T)
    validate_sigma(sigma)
    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}")

    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    drift = (r - q - 0.5 * sigma**2) * T
    vol_t = sigma * np.sqrt(T)

    def payoff(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        st = S * np.exp(drift + vol_t * z)
        pay = np.maximum(st - K, 0.0) if kind == "call" else np.maximum(K - st, 0.0)
        return disc * pay, disc * st

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        pay_p, ctrl_p = payoff(z)
        pay_m, ctrl_m = payoff(-z)
        samples = 0.5 * (pay_p + pay_m)
        controls = 0.5 * (ctrl_p + ctrl_m)
        n_eff = half
    else:
        samples, controls = payoff(rng.standard_normal(n_paths))
        n_eff = n_paths

    if control_variate:
        e_ctrl = S * np.exp(-q * T)
        cov = np.cov(samples, controls, ddof=1)
        var_ctrl = cov[1, 1]
        if var_ctrl > 1e-20:
            beta = cov[0, 1] / var_ctrl
            samples = samples - beta * (controls - e_ctrl)

    est = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n_eff))
    return MCResult(est, se, n_paths, antithetic, control_variate)
