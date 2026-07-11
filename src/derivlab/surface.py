"""Build an implied volatility surface from market option quotes.

Pipeline: raw quotes -> mid prices -> liquidity & no-arbitrage filters ->
vectorized implied vol inversion -> tidy DataFrame -> (T x K) grid.

Standard equity-vol practice is applied: the surface is built from
out-of-the-money options only (puts below spot, calls above), because OTM
contracts carry the volatility information while ITM quotes are dominated
by intrinsic value and wider spreads.

Day-count: ACT/365 (calendar days / 365).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ._validate import validate_kind, validate_positive
from .implied_vol import implied_vol

__all__ = ["OptionQuote", "build_surface", "to_grid"]


@dataclass(frozen=True)
class OptionQuote:
    expiration: date
    strike: float
    kind: str  # "call" | "put"
    bid: float
    ask: float

    def __post_init__(self) -> None:
        validate_kind(self.kind)
        validate_positive("strike", self.strike)
        if self.bid < 0 or self.ask < 0:
            raise ValueError("bid and ask must be >= 0")

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


def build_surface(
    quotes: list[OptionQuote],
    spot: float,
    r: float,
    q: float = 0.0,
    asof: date | None = None,
    otm_only: bool = True,
    min_bid: float = 0.05,
    max_rel_spread: float = 0.5,
    moneyness: tuple[float, float] = (0.80, 1.20),
) -> pd.DataFrame:
    """Return a tidy DataFrame: expiration, T, strike, kind, mid, moneyness, iv.

    Filters applied (each is a data-quality guard, not a modeling choice):
      * bid >= min_bid            — zero-bid quotes are untradeable noise
      * (ask-bid)/mid <= max_rel_spread — wide markets give meaningless mids
      * moneyness window          — deep wings rarely invert cleanly
      * otm_only                  — puts for K < spot, calls for K > spot
      * failed inversions (NaN)   — dropped
    """
    validate_positive("spot", spot)
    asof = asof or date.today()
    kept: list[dict[str, object]] = []
    mids: list[float] = []
    strikes: list[float] = []
    ts: list[float] = []
    kinds: list[str] = []

    for qt in quotes:
        t = (qt.expiration - asof).days / 365.0
        if t <= 0:
            continue
        m = qt.strike / spot
        if not (moneyness[0] <= m <= moneyness[1]):
            continue
        if otm_only:
            if qt.kind == "put" and qt.strike > spot:
                continue
            if qt.kind == "call" and qt.strike < spot:
                continue
        if qt.bid < min_bid or qt.ask <= 0 or qt.ask < qt.bid:
            continue
        mid = qt.mid
        if mid <= 0 or (qt.ask - qt.bid) / mid > max_rel_spread:
            continue
        kept.append(
            {
                "expiration": qt.expiration,
                "T": t,
                "strike": qt.strike,
                "kind": qt.kind,
                "mid": mid,
                "moneyness": m,
            }
        )
        mids.append(mid)
        strikes.append(qt.strike)
        ts.append(t)
        kinds.append(qt.kind)

    if not kept:
        return pd.DataFrame(
            columns=["expiration", "T", "strike", "kind", "mid", "moneyness", "iv"]
        )

    ivs = np.full(len(kept), np.nan)
    for kind in ("call", "put"):
        mask = np.array([k == kind for k in kinds])
        if not np.any(mask):
            continue
        iv_kind = implied_vol(
            np.array(mids)[mask],
            spot,
            np.array(strikes)[mask],
            np.array(ts)[mask],
            r,
            q,
            kind,
        )
        ivs[mask] = np.asarray(iv_kind, dtype=float)

    df = pd.DataFrame(kept)
    df["iv"] = ivs
    df = df.dropna(subset=["iv"])
    if not df.empty:
        df = df.sort_values(["T", "strike"]).reset_index(drop=True)
    return df


def to_grid(surface: pd.DataFrame) -> pd.DataFrame:
    """Pivot the tidy surface into a (T x strike) grid of implied vols.

    Where both an OTM put and call exist at the same (T, K) — near the
    money — their vols are averaged.
    """
    return surface.pivot_table(index="T", columns="strike", values="iv", aggfunc="mean")
