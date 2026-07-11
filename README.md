# derivatives-lab

Options pricing, Greeks, and implied volatility surfaces in Python — with
**three pricing engines that cross-validate each other** and a surface
pipeline validated against **live SPY option chains** and the broker's own
published implied vols.

```
IV solver vs broker IVs   0.52 vol pts mean abs diff (76 live SPY contracts)
surface round-trip        recovers a known smile to < 1e-6
test suite                33 tests: parity, convergence, variance reduction, SPY integration
```

## What's inside

| Module | Contents |
|---|---|
| `derivlab.bs` | Black-Scholes-Merton analytic prices + full Greeks (vectorized) |
| `derivlab.binomial` | CRR tree, European & American exercise, vectorized backward induction |
| `derivlab.monte_carlo` | GBM Monte Carlo with antithetic + control variates; every price carries a standard error |
| `derivlab.implied_vol` | Newton + Brent-fallback inversion with no-arbitrage rejection |
| `derivlab.surface` | Market quotes → filtered, OTM-only implied vol surface |
| `derivlab.viz` | Interactive 3D surface and smile plots (Plotly) |

## Validation philosophy

A pricing library you can't verify is a random number generator with
confidence. The test suite is organized the way a model-validation team
would sign off:

1. **Analytic engine vs textbook** — reproduces Hull's published values.
2. **Static no-arbitrage** — put-call parity holds to 1e-9 across 200
   random parameter sets; Greek identities (gamma/vega call-put equality,
   delta bounds, `Δc − Δp = e^{−qT}`).
3. **Greeks vs finite differences** — every analytic Greek is checked
   against a numerical derivative of the price function.
4. **Cross-engine convergence** — the binomial tree converges to BS for
   European options; Monte Carlo lands within its own confidence interval
   of the analytic price.
5. **American exercise** — deep-ITM puts carry an early-exercise premium;
   American calls on non-dividend stock equal European (Merton).
6. **Variance reduction reduces variance** — antithetic + control variates
   are verified to cut the standard error by more than half, not assumed to.
7. **Surface round-trip** — a synthetic chain priced with a known smile is
   rebuilt by the pipeline and recovers that smile to < 1e-6.

```
33 passed in 1.69s
```

## Live data: the SPY surface

`examples/build_surface.py` builds the surface from a real snapshot —
80 OTM SPY option quotes across 5 expirations (Jul 2026 → Dec 2026),
captured at the 4:15pm ET options close on 2026-07-09 via a read-only
Robinhood market-data connection. The pipeline's liquidity filters
correctly reject the four near-worthless wing contracts; the remaining 76
invert cleanly.

**Independent validation:** the broker publishes its own per-contract IVs.
derivlab's European BSM inversion agrees with them to a **mean absolute
difference of 0.52 vol points** (max 1.33) — the residual is explained by
their American-exercise model and discrete-dividend handling vs our
continuous-yield approximation, both documented assumptions.

The captured ATM term structure (upward-sloping, calm regime):

| Expiry | T (yrs) | ATM IV |
|---|---|---|
| 2026-07-17 | 0.02 | 10.8% |
| 2026-08-21 | 0.12 | 13.6% |
| 2026-09-18 | 0.20 | 14.7% |
| 2026-10-16 | 0.27 | 15.2% |
| 2026-12-18 | 0.44 | 16.6% |

The put skew is visible in the raw data: 690-strike puts trade above 20%
IV at every tenor while ATM sits at 11-17% — the market pays up for crash
protection, the defining feature of equity index vol since 1987.

```bash
python3 examples/build_surface.py
# -> examples/spy_surface.html (interactive 3D surface, ~100 KB with CDN)
# -> examples/spy_smiles.html  (per-expiry smiles)
```

## Install & test

```bash
pip install -e ".[dev,viz]"
pytest
ruff check src tests examples
mypy
```

Core pricing requires only NumPy, SciPy, and pandas. Plotly is optional:

```bash
pip install -e .              # pricing only
pip install -e ".[viz]"       # + Plotly charts
```

Pure Python. No API keys required — the live SPY snapshot ships with the repo;
re-pulling fresh chains needs any option-quote source you have access to.

## Documentation

- [API reference](docs/api.md)
- [Conventions & assumptions](docs/conventions.md)
- [Changelog](CHANGELOG.md)

## Design notes

- **Every MC price carries a standard error.** A Monte Carlo estimate
  without an error bar is meaningless; `MCResult.ci()` gives the interval.
- **Antithetic SE is computed over pair-averages** — the statistically
  correct estimator, not the common off-by-√2 mistake.
- **The IV solver returns NaN, not garbage,** for prices violating static
  no-arbitrage bounds — essential for real quotes, where wings routinely
  contain stale or crossed markets.
- **OTM-only surface construction** (puts below spot, calls above) —
  standard equity-vol practice, since ITM quotes are dominated by intrinsic
  value and wider spreads.

## Roadmap

- SVI smile parameterization + calendar/butterfly arbitrage checks
- American-exercise IV inversion (de-Americanization via the binomial engine)
- Discrete dividend handling for single names
- Greeks surfaces and a scenario ladder (spot × vol P&L grid)
- Integration with `quant-engine` as the pricing layer

## License

MIT
