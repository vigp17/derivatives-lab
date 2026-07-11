"""Integration test against the bundled SPY option chain snapshot."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from derivlab import OptionQuote, build_surface

HERE = Path(__file__).parent
SNAPSHOT = HERE.parent / "examples" / "data" / "spy_chain_2026-07-09.json"
R, Q = 0.036, 0.011


def test_spy_surface_matches_broker_ivs():
    snap = json.loads(SNAPSHOT.read_text())
    spot = snap["spot"]
    asof = date.fromisoformat(snap["asof"])

    quotes, rh_iv = [], {}
    for q_ in snap["quotes"]:
        key = (q_["expiration"], q_["strike"], q_["kind"])
        quotes.append(
            OptionQuote(
                date.fromisoformat(q_["expiration"]),
                float(q_["strike"]),
                q_["kind"],
                q_["bid"],
                q_["ask"],
            )
        )
        if q_["rh_iv"] is not None:
            rh_iv[key] = q_["rh_iv"]

    surf = build_surface(quotes, spot=spot, r=R, q=Q, asof=asof)
    assert len(surf) == 76

    diffs = []
    for row in surf.itertuples():
        key = (row.expiration.isoformat(), row.strike, row.kind)
        if key in rh_iv:
            diffs.append((row.iv - rh_iv[key]) * 100)

    diffs = np.array(diffs)
    assert len(diffs) == 76
    assert np.mean(np.abs(diffs)) == pytest.approx(0.52, abs=0.05)
    assert np.max(np.abs(diffs)) == pytest.approx(1.33, abs=0.15)
