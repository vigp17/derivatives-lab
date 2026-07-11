"""Visualization: interactive 3D vol surface and per-expiry smiles (Plotly).

Requires the optional ``derivlab[viz]`` extra (``pip install derivlab[viz]``).
"""

from __future__ import annotations

import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - exercised via lazy import tests
    raise ImportError(
        "Plotly is required for derivlab.viz. Install with: pip install derivlab[viz]"
    ) from exc

__all__ = ["surface_figure", "smile_figure"]


def surface_figure(
    grid: pd.DataFrame, title: str = "Implied Volatility Surface"
) -> go.Figure:
    """3D surface from a (T x strike) IV grid (as produced by surface.to_grid)."""
    strikes = grid.columns.to_numpy(dtype=float)
    expiries = grid.index.to_numpy(dtype=float)
    z = grid.to_numpy(dtype=float) * 100.0

    fig = go.Figure(
        go.Surface(
            x=strikes,
            y=expiries,
            z=z,
            colorscale="Viridis",
            colorbar=dict(title="IV (%)"),
            connectgaps=True,
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Time to expiry (years)",
            zaxis_title="Implied vol (%)",
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.7)),
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def smile_figure(
    surface: pd.DataFrame, spot: float, title: str = "Volatility Smiles by Expiry"
) -> go.Figure:
    """Overlay the smile (IV vs strike) for each expiration in a tidy surface."""
    fig = go.Figure()
    for exp, sub in surface.groupby("expiration"):
        sub = sub.sort_values("strike")
        fig.add_trace(
            go.Scatter(
                x=sub["strike"],
                y=sub["iv"] * 100.0,
                mode="lines+markers",
                name=str(exp),
            )
        )
    fig.add_vline(x=spot, line_dash="dash", line_color="gray", annotation_text="spot")
    fig.update_layout(
        title=title,
        xaxis_title="Strike",
        yaxis_title="Implied vol (%)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
