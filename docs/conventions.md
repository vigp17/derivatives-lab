# Conventions

This document centralizes modeling assumptions used across `derivlab`.

## Day count

- Time to expiry `T` is measured in **years** using **ACT/365**: calendar days between
  `asof` and expiration divided by 365.
- Used in `surface.build_surface` and all pricing engines.

## Rates and dividends

- `r` — continuously compounded risk-free rate.
- `q` — continuous dividend yield (equity index approximation).
- No discrete cash dividends in v0.1.0 (roadmap item for single names).

## Volatility and Greeks

- `sigma` — annualized volatility, must be strictly positive.
- **Vega** and **rho** are reported per **unit** change in volatility and rate.
  Multiply by `0.01` for per-1% sensitivities.
- **Theta** is per calendar year (same units as `T`).

## Implied volatility

- European BSM inversion only (American/de-Americanization is on the roadmap).
- Prices outside static no-arbitrage bounds return `NaN` (strict inequality: `lo < price < hi`).
- Solver: Newton-Raphson on vega, Brent bracketing fallback.

## Surface construction

- **OTM-only** by default: puts with `K <= spot`, calls with `K >= spot`.
- Liquidity filters: minimum bid, maximum relative spread, moneyness window.
- Near-the-money duplicate `(T, K)` pairs are averaged in `to_grid`.

## Monte Carlo

- GBM under the risk-neutral measure with constant `r`, `q`, `sigma`.
- Standard error computed over **antithetic pair averages** when antithetic is enabled.
- Control variate uses discounted terminal stock price; skipped when control variance is zero.

## Example assumptions (SPY snapshot)

| Parameter | Value | Source |
|---|---|---|
| `r` | 3.6% | Short-term Treasury yield at snapshot date |
| `q` | 1.1% | SPY trailing dividend yield (continuous approx.) |
| Day count | ACT/365 | Calendar days / 365 |

These are documented in `examples/build_surface.py`, not hidden in the library defaults.
