# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-07-10

### Added
- Black-Scholes-Merton analytic pricing and Greeks (`derivlab.bs`)
- CRR binomial tree with European and American exercise (`derivlab.binomial`)
- Monte Carlo pricing with antithetic variates and control variates (`derivlab.monte_carlo`)
- Implied volatility inversion with Newton + Brent fallback (`derivlab.implied_vol`)
- IV surface pipeline from market quotes (`derivlab.surface`)
- Plotly visualization helpers (`derivlab.viz`, optional extra)
- 33+ validation tests including cross-engine convergence and SPY integration
- GitHub Actions CI (ruff, mypy, pytest, pip-audit)
- Documentation: conventions, API reference, CHANGELOG

### Changed
- Plotly moved to optional `derivlab[viz]` extra
- Vectorized implied vol inversion for batch surface builds
- Input validation across all pricing engines

[0.1.0]: https://github.com/vigp17/derivatives-lab/releases/tag/v0.1.0
