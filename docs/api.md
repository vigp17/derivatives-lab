# API Reference

## Package: `derivlab`

```python
import derivlab
from derivlab import bs, binomial, monte_carlo, implied_vol
from derivlab import OptionQuote, build_surface, to_grid
from derivlab.viz import surface_figure, smile_figure  # requires derivlab[viz]
```

---

## `derivlab.bs`

### `price(S, K, T, r, sigma, q=0.0, kind="call")`

Black-Scholes-Merton European option price. Accepts scalars or NumPy arrays.

- **Returns:** `float` or `ndarray`
- **Expiry:** `T <= 0` returns intrinsic value

### `greeks(S, K, T, r, sigma, q=0.0, kind="call")`

Analytic Greeks: `delta`, `gamma`, `vega`, `theta`, `rho`.

- **Requires:** `T > 0`, `S > 0`, `K > 0`, `sigma > 0`

### `d1_d2(S, K, T, r, sigma, q=0.0)`

BSM `d1` and `d2` terms.

---

## `derivlab.binomial`

### `price(S, K, T, r, sigma, q=0.0, kind="call", style="european", steps=800)`

CRR binomial tree price.

- **style:** `"european"` or `"american"`
- **Expiry:** `T <= 0` returns intrinsic value

---

## `derivlab.monte_carlo`

### `MCResult`

Frozen dataclass: `price`, `std_err`, `n_paths`, `antithetic`, `control_variate`.

- **`ci(z=1.96)`** — approximate confidence interval

### `price(S, K, T, r, sigma, q=0.0, kind="call", n_paths=200_000, antithetic=True, control_variate=True, seed=None)`

Monte Carlo price under GBM with optional variance reduction.

---

## `derivlab.implied_vol`

### `implied_vol(target_price, S, K, T, r, q=0.0, kind="call", tol=1e-10, max_iter=60)`

Invert BSM to implied volatility. Accepts scalars or arrays.

- **Returns:** `float` or `ndarray` of vols; `NaN` for no-arb violations or `T <= 0`

---

## `derivlab.surface`

### `OptionQuote(expiration, strike, kind, bid, ask)`

Frozen dataclass for a single market quote.

### `build_surface(quotes, spot, r, q=0.0, asof=None, otm_only=True, min_bid=0.05, max_rel_spread=0.5, moneyness=(0.80, 1.20))`

Build a tidy IV surface DataFrame with columns:
`expiration`, `T`, `strike`, `kind`, `mid`, `moneyness`, `iv`.

### `to_grid(surface)`

Pivot to a `(T × strike)` IV grid; averages duplicate `(T, K)` entries.

---

## `derivlab.viz` (optional)

Requires `pip install derivlab[viz]`.

### `surface_figure(grid, title="Implied Volatility Surface")`

Plotly 3D surface from `to_grid` output.

### `smile_figure(surface, spot, title="Volatility Smiles by Expiry")`

Plotly 2D smile overlay by expiration.

---

See also: [Conventions](conventions.md)
