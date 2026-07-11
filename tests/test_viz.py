"""Smoke tests for Plotly visualization helpers."""

from datetime import date

import pandas as pd
import plotly.graph_objects as go

from derivlab.viz import smile_figure, surface_figure


def test_surface_figure_returns_plotly_figure():
    grid = pd.DataFrame(
        [[0.18, 0.20], [0.19, 0.21]],
        index=[0.1, 0.2],
        columns=[95.0, 100.0],
    )
    fig = surface_figure(grid, title="test surface")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "test surface"


def test_smile_figure_returns_plotly_figure():
    exp = date(2026, 8, 1)
    surface = pd.DataFrame(
        {
            "expiration": [exp, exp],
            "T": [0.1, 0.1],
            "strike": [95.0, 100.0],
            "kind": ["put", "call"],
            "mid": [1.0, 1.0],
            "moneyness": [0.95, 1.0],
            "iv": [0.22, 0.20],
        }
    )
    fig = smile_figure(surface, spot=100.0, title="test smile")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_viz_lazy_import():
    import derivlab

    viz = derivlab.viz
    assert hasattr(viz, "surface_figure")
