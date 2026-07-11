"""Build the SPY implied volatility surface from a live chain snapshot.

The snapshot in data/ was captured via the Robinhood MCP connector
(read-only market data) at the 4:15pm ET options close. This script:
  1. loads the raw quotes and runs them through derivlab's surface pipeline
  2. validates derivlab's implied vols against the broker's own published
     IVs, contract by contract
  3. renders an interactive 3D surface and per-expiry smiles (Plotly HTML)

Run from the repo root:  python3 examples/build_surface.py
"""

import json
from datetime import date
from pathlib import Path

import numpy as np

from derivlab import OptionQuote, build_surface, to_grid
from derivlab.viz import smile_figure, surface_figure

HERE = Path(__file__).parent
SNAPSHOT = HERE / "data" / "spy_chain_2026-07-09.json"

# Rate/dividend assumptions (documented, not hidden):
#   r — short-term Treasury yield at the snapshot date
#   q — SPY trailing dividend yield, continuous approximation
R, Q = 0.036, 0.011


def main() -> None:
    snap = json.loads(SNAPSHOT.read_text())
    spot = snap["spot"]
    asof = date.fromisoformat(snap["asof"])

    quotes, rh_iv = [], {}
    for q_ in snap["quotes"]:
        key = (q_["expiration"], q_["strike"], q_["kind"])
        quotes.append(OptionQuote(date.fromisoformat(q_["expiration"]),
                                  float(q_["strike"]), q_["kind"],
                                  q_["bid"], q_["ask"]))
        if q_["rh_iv"] is not None:
            rh_iv[key] = q_["rh_iv"]

    surf = build_surface(quotes, spot=spot, r=R, q=Q, asof=asof)
    print(f"{snap['symbol']} {snap['asof']}  spot={spot}  "
          f"{len(surf)}/{len(quotes)} quotes survived filters\n")

    # ---- validation vs broker IVs -----------------------------------------
    diffs = []
    for row in surf.itertuples():
        key = (row.expiration.isoformat(), row.strike, row.kind)
        if key in rh_iv:
            diffs.append((row.iv - rh_iv[key]) * 100)  # vol points
    diffs = np.array(diffs)
    print("derivlab IV vs Robinhood published IV "
          f"({len(diffs)} contracts, vol points):")
    print(f"  mean abs diff : {np.mean(np.abs(diffs)):.3f}")
    print(f"  median        : {np.median(diffs):+.3f}")
    print(f"  max abs       : {np.max(np.abs(diffs)):.3f}\n")

    # ---- term structure at the money ---------------------------------------
    print("ATM-ish IV term structure (750/755 strikes):")
    atm = surf[surf.strike.isin([750.0, 755.0])]
    for exp, sub in atm.groupby("expiration"):
        print(f"  {exp}  T={sub['T'].iloc[0]:.3f}y  "
              f"iv={sub['iv'].mean() * 100:.2f}%")

    # ---- figures ------------------------------------------------------------
    grid = to_grid(surf)
    fig3d = surface_figure(grid, title=f"SPY IV Surface — {snap['asof']} close")
    fig3d.write_html(HERE / "spy_surface.html", include_plotlyjs="cdn")
    smiles = smile_figure(surf, spot, title=f"SPY Smiles by Expiry — {snap['asof']}")
    smiles.write_html(HERE / "spy_smiles.html", include_plotlyjs="cdn")
    print(f"\nwrote {HERE / 'spy_surface.html'}")
    print(f"wrote {HERE / 'spy_smiles.html'}")


if __name__ == "__main__":
    main()
