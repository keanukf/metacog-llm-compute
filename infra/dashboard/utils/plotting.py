"""Plotly chart builders for calibration and analysis."""
from __future__ import annotations

from typing import Any

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def reliability_diagram(
    bin_centers: list[float],
    mean_confidence: list[float],
    mean_accuracy: list[float],
    title: str = "Reliability diagram",
) -> Any:
    """Build a reliability diagram (confidence vs accuracy per bin)."""
    if not _AVAILABLE:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mean_confidence,
            y=mean_accuracy,
            mode="lines+markers",
            name="Model",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect",
            line=dict(color="gray", dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Mean predicted confidence",
        yaxis_title="Mean accuracy",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=True,
    )
    return fig


def success_rate_bars(
    labels: list[str],
    success_rates: list[float],
    title: str = "Success rate by condition",
) -> Any:
    """Bar chart of success rates."""
    if not _AVAILABLE:
        return None
    fig = go.Figure(
        data=[go.Bar(x=labels, y=success_rates, marker_color="steelblue")]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Condition",
        yaxis_title="Success rate",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def efficiency_plot(
    strategies: list[str],
    success_rates: list[float],
    normalized_costs: list[float],
    title: str = "Efficiency: success rate vs compute cost",
) -> Any:
    """Scatter: x=normalized cost, y=success rate, one point per strategy."""
    if not _AVAILABLE:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=normalized_costs,
            y=success_rates,
            mode="markers+text",
            text=strategies,
            textposition="top center",
            marker=dict(size=12),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Normalized compute cost",
        yaxis_title="Success rate",
        showlegend=False,
    )
    return fig


def metric_over_steps(
    steps: list[int],
    values: list[float],
    name: str = "Metric",
    title: str = "Metric over episodes",
) -> Any:
    """Line chart of a metric vs step index."""
    if not _AVAILABLE:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=steps, y=values, mode="lines+markers", name=name)
    )
    fig.update_layout(
        title=title,
        xaxis_title="Episode step",
        yaxis_title=name,
    )
    return fig
