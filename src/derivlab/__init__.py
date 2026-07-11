"""derivlab — options pricing, Greeks, and implied volatility surfaces.

Engines: Black-Scholes analytic (bs), CRR binomial tree (binomial),
Monte Carlo with variance reduction (monte_carlo). Inversion via
implied_vol; surface construction in surface; Plotly charts in viz
(optional extra).
"""

from __future__ import annotations

from typing import Any

from . import binomial, bs, monte_carlo, surface
from .implied_vol import implied_vol
from .surface import OptionQuote, build_surface, to_grid

__version__ = "0.1.0"
__all__ = [
    "bs",
    "binomial",
    "monte_carlo",
    "surface",
    "viz",
    "implied_vol",
    "OptionQuote",
    "build_surface",
    "to_grid",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name == "viz":
        from . import viz

        return viz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
