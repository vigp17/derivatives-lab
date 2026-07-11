"""Validation suite for derivlab.

Structure mirrors how a model-validation team would sign off a pricing
library:
  1. Analytic engine vs published textbook values (Hull)
  2. Static no-arbitrage identities (put-call parity, Greek bounds)
  3. Greeks vs finite differences of the price function
  4. Cross-engine convergence: binomial -> BS, Monte Carlo -> BS
  5. American-exercise properties
  6. Variance reduction actually reduces variance
  7. Implied vol round-trips and rejects arbitrage-violating prices
  8. End-to-end surface round-trip: price a synthetic chain with a known
     smile, rebuild the surface, recover the smile.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from derivlab import binomial, bs, monte_carlo
from derivlab.implied_vol import implied_vol
from derivlab.surface import OptionQuote, build_surface, to_grid

# Hull, "Options, Futures, and Other Derivatives": S=42, K=40, r=10%,
# sigma=20%, T=0.5 -> call ~ 4.76, put ~ 0.81.
HULL = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)


# ------------------------------------------------------------- analytic BS

def test_bs_matches_hull_textbook_values():
    assert bs.price(**HULL, kind="call") == pytest.approx(4.76, abs=0.01)
    assert bs.price(**HULL, kind="put") == pytest.approx(0.81, abs=0.01)


def test_put_call_parity_across_random_parameters():
    rng = np.random.default_rng(3)
    for _ in range(200):
        S = rng.uniform(20, 400)
        K = S * rng.uniform(0.6, 1.4)
        T = rng.uniform(0.05, 3.0)
        r = rng.uniform(0.0, 0.08)
        q = rng.uniform(0.0, 0.04)
        sigma = rng.uniform(0.08, 0.9)
        c = bs.price(S, K, T, r, sigma, q, "call")
        p = bs.price(S, K, T, r, sigma, q, "put")
        lhs = c - p
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert lhs == pytest.approx(rhs, abs=1e-9)


def test_bs_expiry_returns_intrinsic():
    assert bs.price(105, 100, 0.0, 0.05, 0.2, kind="call") == pytest.approx(5.0)
    assert bs.price(95, 100, 0.0, 0.05, 0.2, kind="put") == pytest.approx(5.0)


def test_bs_vectorized_over_strikes():
    strikes = np.array([80.0, 100.0, 120.0])
    prices = bs.price(100.0, strikes, 1.0, 0.05, 0.2, kind="call")
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)


def test_bs_rejects_non_positive_inputs():
    with pytest.raises(ValueError, match="sigma must be > 0"):
        bs.price(100, 100, 1.0, 0.05, 0.0, kind="call")
    with pytest.raises(ValueError, match="S and K must be > 0"):
        bs.d1_d2(0, 100, 1.0, 0.05, 0.2)


def test_greeks_require_positive_time():
    with pytest.raises(ValueError, match="Greeks require T > 0"):
        bs.greeks(100, 100, 0.0, 0.05, 0.2, kind="call")


# ------------------------------------------------------------------ greeks

def test_greeks_match_finite_differences():
    S, K, T, r, sigma, q = 100.0, 105.0, 0.75, 0.04, 0.25, 0.01
    for kind in ("call", "put"):
        g = bs.greeks(S, K, T, r, sigma, q, kind)
        h = 1e-4
        fd_delta = (bs.price(S + h, K, T, r, sigma, q, kind)
                    - bs.price(S - h, K, T, r, sigma, q, kind)) / (2 * h)
        fd_gamma = (bs.price(S + h, K, T, r, sigma, q, kind)
                    - 2 * bs.price(S, K, T, r, sigma, q, kind)
                    + bs.price(S - h, K, T, r, sigma, q, kind)) / h**2
        fd_vega = (bs.price(S, K, T, r, sigma + h, q, kind)
                   - bs.price(S, K, T, r, sigma - h, q, kind)) / (2 * h)
        fd_theta = -(bs.price(S, K, T + h, r, sigma, q, kind)
                     - bs.price(S, K, T - h, r, sigma, q, kind)) / (2 * h)
        fd_rho = (bs.price(S, K, T, r + h, sigma, q, kind)
                  - bs.price(S, K, T, r - h, sigma, q, kind)) / (2 * h)
        assert g["delta"] == pytest.approx(fd_delta, abs=1e-6)
        assert g["gamma"] == pytest.approx(fd_gamma, abs=1e-4)
        assert g["vega"] == pytest.approx(fd_vega, abs=1e-4)
        assert g["theta"] == pytest.approx(fd_theta, abs=1e-4)
        assert g["rho"] == pytest.approx(fd_rho, abs=1e-4)


def test_greek_identities_and_bounds():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.2
    gc = bs.greeks(S, K, T, r, sigma, kind="call")
    gp = bs.greeks(S, K, T, r, sigma, kind="put")
    assert gc["gamma"] == pytest.approx(gp["gamma"], rel=1e-12)
    assert gc["vega"] == pytest.approx(gp["vega"], rel=1e-12)
    assert 0.0 <= gc["delta"] <= 1.0
    assert -1.0 <= gp["delta"] <= 0.0
    assert gc["delta"] - gp["delta"] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------- binomial

def test_binomial_converges_to_bs_european():
    for kind in ("call", "put"):
        b = binomial.price(**HULL, kind=kind, style="european", steps=2000)
        a = bs.price(**HULL, kind=kind)
        assert b == pytest.approx(a, abs=5e-3)


def test_american_put_carries_early_exercise_premium():
    kw = dict(S=80.0, K=100.0, T=1.0, r=0.08, sigma=0.2)
    eu = binomial.price(**kw, kind="put", style="european", steps=1000)
    am = binomial.price(**kw, kind="put", style="american", steps=1000)
    assert am > eu + 1e-3
    assert am >= 100.0 - 80.0


def test_american_call_no_dividends_equals_european():
    kw = dict(S=100.0, K=95.0, T=1.0, r=0.05, sigma=0.3, q=0.0)
    eu = binomial.price(**kw, kind="call", style="european", steps=1000)
    am = binomial.price(**kw, kind="call", style="american", steps=1000)
    assert am == pytest.approx(eu, abs=1e-9)


def test_binomial_expiry_returns_intrinsic():
    assert binomial.price(110, 100, 0.0, 0.05, 0.2, kind="call") == pytest.approx(10.0)


def test_binomial_rejects_invalid_steps():
    with pytest.raises(ValueError, match="steps must be > 0"):
        binomial.price(100, 100, 1.0, 0.05, 0.2, steps=0)


# ------------------------------------------------------------- monte carlo

def test_mc_price_within_confidence_interval_of_bs():
    a = bs.price(**HULL, kind="call")
    res = monte_carlo.price(**HULL, kind="call", n_paths=400_000, seed=11)
    assert abs(res.price - a) < 4.0 * res.std_err
    lo, hi = res.ci()
    assert lo < hi


def test_variance_reduction_reduces_standard_error():
    kw = dict(**HULL, kind="call", n_paths=200_000, seed=5)
    plain = monte_carlo.price(**kw, antithetic=False, control_variate=False)
    anti = monte_carlo.price(**kw, antithetic=True, control_variate=False)
    full = monte_carlo.price(**kw, antithetic=True, control_variate=True)
    assert anti.std_err < plain.std_err
    assert full.std_err < anti.std_err
    assert full.std_err < 0.5 * plain.std_err


def test_mc_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="n_paths must be > 0"):
        monte_carlo.price(100, 100, 1.0, 0.05, 0.2, n_paths=0)


# -------------------------------------------------------------- implied vol

def test_implied_vol_round_trip():
    for kind in ("call", "put"):
        for sigma in (0.08, 0.23, 0.65, 1.4):
            px = bs.price(100, 110, 0.6, 0.04, sigma, 0.01, kind)
            iv = implied_vol(px, 100, 110, 0.6, 0.04, 0.01, kind)
            assert iv == pytest.approx(sigma, abs=1e-7)


def test_implied_vol_vectorized_round_trip():
    sigmas = np.array([0.12, 0.25, 0.40])
    mids = bs.price(100.0, np.array([95.0, 100.0, 105.0]), 0.5, 0.04, sigmas, kind="call")
    ivs = implied_vol(mids, 100.0, np.array([95.0, 100.0, 105.0]), 0.5, 0.04, kind="call")
    assert np.allclose(ivs, sigmas, atol=1e-7)


def test_implied_vol_rejects_arbitrage_violations():
    assert np.isnan(implied_vol(0.5, 100, 80, 0.5, 0.05, kind="call"))
    assert np.isnan(implied_vol(101.0, 100, 80, 0.5, 0.05, kind="call"))
    assert np.isnan(implied_vol(5.0, 100, 100, 0.0, 0.05, kind="call"))


def test_implied_vol_extreme_vol_round_trip():
    sigma = 2.5
    px = bs.price(100, 100, 0.25, 0.03, sigma, kind="call")
    assert implied_vol(px, 100, 100, 0.25, 0.03, kind="call") == pytest.approx(sigma, abs=1e-6)


# ------------------------------------------------------------------ surface

def _smile(strike: float, spot: float, t: float) -> float:
    m = np.log(strike / spot)
    return 0.20 - 0.15 * m / np.sqrt(t) + 0.4 * m**2 / t**0.5


def test_surface_round_trip_recovers_known_smile():
    spot, r, q = 500.0, 0.04, 0.012
    asof = date(2026, 7, 6)
    quotes = []
    for days in (30, 60, 120):
        exp = asof + timedelta(days=days)
        t = days / 365.0
        for strike in np.arange(440, 565, 5.0):
            vol = _smile(strike, spot, t)
            kind = "put" if strike <= spot else "call"
            px = bs.price(spot, strike, t, r, vol, q, kind)
            quotes.append(OptionQuote(exp, float(strike), kind,
                                      bid=px - 0.02, ask=px + 0.02))
    surf = build_surface(quotes, spot, r, q, asof=asof)
    assert len(surf) > 50
    err = [abs(row.iv - _smile(row.strike, spot, row.T)) for row in surf.itertuples()]
    assert max(err) < 1e-6

    grid = to_grid(surf)
    assert grid.shape[0] == 3
    assert grid.notna().sum().sum() == len(surf)


def test_surface_filters_bad_quotes():
    spot, r = 500.0, 0.04
    asof = date(2026, 7, 6)
    exp = asof + timedelta(days=30)
    quotes = [
        OptionQuote(exp, 480.0, "put", bid=0.0, ask=0.10),
        OptionQuote(exp, 490.0, "put", bid=1.00, ask=0.50),
        OptionQuote(exp, 495.0, "put", bid=0.10, ask=5.00),
        OptionQuote(exp, 520.0, "put", bid=21.0, ask=21.5),
        OptionQuote(exp, 300.0, "put", bid=0.10, ask=0.12),
    ]
    surf = build_surface(quotes, spot, r, asof=asof)
    assert surf.empty


def test_option_quote_validates_kind_and_strike():
    with pytest.raises(ValueError, match="kind must be"):
        OptionQuote(date.today(), 100.0, "invalid", 1.0, 1.1)
    with pytest.raises(ValueError, match="strike must be > 0"):
        OptionQuote(date.today(), 0.0, "call", 1.0, 1.1)


def test_build_surface_rejects_non_positive_spot():
    exp = date(2026, 7, 6) + timedelta(days=30)
    quotes = [OptionQuote(exp, 100.0, "call", 1.0, 1.1)]
    with pytest.raises(ValueError, match="spot must be > 0"):
        build_surface(quotes, spot=0.0, r=0.04)


def test_package_lazy_viz_and_version():
    import derivlab

    assert derivlab.__version__ == "0.1.0"
    assert hasattr(derivlab.viz, "surface_figure")
    with pytest.raises(AttributeError):
        _ = derivlab.nope  # type: ignore[attr-defined]


def test_to_grid_averages_duplicate_strikes():
    import pandas as pd

    surface = pd.DataFrame(
        {
            "T": [0.25, 0.25],
            "strike": [100.0, 100.0],
            "iv": [0.20, 0.22],
            "expiration": [date(2026, 10, 1), date(2026, 10, 1)],
            "kind": ["put", "call"],
            "mid": [1.0, 1.0],
            "moneyness": [1.0, 1.0],
        }
    )
    grid = to_grid(surface)
    assert grid.loc[0.25, 100.0] == pytest.approx(0.21)
