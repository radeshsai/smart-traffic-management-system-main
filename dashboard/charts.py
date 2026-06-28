"""
dashboard/charts.py — Plotly Chart Builders
============================================
All chart/visualization functions for the Streamlit dashboard.
Returns plotly Figure objects ready for st.plotly_chart().
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Optional

# Brand colours per direction
DIR_COLORS = {
    "north": "#4FC3F7",
    "south": "#81C784",
    "east":  "#FF8A65",
    "west":  "#CE93D8",
}

# Density level colours
DENSITY_COLORS = {
    "LOW":    "#66BB6A",
    "MEDIUM": "#FFA726",
    "HIGH":   "#EF5350",
}

DIRECTIONS = ["north", "south", "east", "west"]


def vehicle_count_bar(counts: Dict[str, int]) -> go.Figure:
    """Bar chart of current vehicle counts per direction."""
    dirs   = list(counts.keys())
    values = list(counts.values())
    colors = [DIR_COLORS.get(d, "#90CAF9") for d in dirs]

    fig = go.Figure(go.Bar(
        x=[d.capitalize() for d in dirs],
        y=values,
        marker_color=colors,
        text=values,
        textposition="outside",
        textfont=dict(size=14, color="white"),
    ))
    fig.update_layout(
        title="🚗 Current Vehicle Count per Direction",
        xaxis_title="Direction",
        yaxis_title="Vehicle Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.8)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        height=320,
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


def density_timeseries(df: pd.DataFrame) -> go.Figure:
    """Line chart of vehicle counts over time per direction."""
    fig = go.Figure()

    for direction in DIRECTIONS:
        sub = df[df["direction"] == direction] if "direction" in df.columns else pd.DataFrame()
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["created_at"],
            y=sub["vehicle_count"],
            mode="lines",
            name=direction.capitalize(),
            line=dict(color=DIR_COLORS.get(direction, "#fff"), width=2),
            fill="tozeroy",
            fillcolor="rgba(255,255,255,0.05)",
        ))

    fig.update_layout(
        title="📈 Vehicle Count Over Time",
        xaxis_title="Time",
        yaxis_title="Vehicles",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.8)",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=350,
        margin=dict(t=60, b=40, l=40, r=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig


def congestion_gauge(score: float, direction: str) -> go.Figure:
    """Gauge chart showing congestion score 0-100 for one direction."""
    color = "#66BB6A" if score < 33 else "#FFA726" if score < 66 else "#EF5350"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(color="white", size=28)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="white", tickfont=dict(color="white")),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(20,20,30,0.8)",
            bordercolor="rgba(255,255,255,0.2)",
            steps=[
                dict(range=[0,  33], color="rgba(102,187,106,0.15)"),
                dict(range=[33, 66], color="rgba(255,167, 38,0.15)"),
                dict(range=[66,100], color="rgba(239, 83, 80,0.15)"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.75,
                value=score,
            ),
        ),
        title=dict(text=direction.upper(), font=dict(color="white", size=14)),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=220,
        margin=dict(t=30, b=10, l=20, r=20),
    )
    return fig


def signal_phase_timeline(cycle_logs: List[Dict]) -> go.Figure:
    """Bar chart of green time allocated per direction."""
    if not cycle_logs:
        fig = go.Figure()
        fig.update_layout(
            title="🚦 Signal Phase Timeline (No data yet)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=200,
        )
        return fig

    df = pd.DataFrame(cycle_logs)

    # Normalise column names:
    # DB summary query  -> uses 'avg_green'
    # Live cycle logs   -> uses 'green_time_allocated'
    if "avg_green" in df.columns and "green_time_allocated" not in df.columns:
        df["green_time_allocated"] = df["avg_green"].round(1)

    # Ensure required columns exist before plotting
    if "green_time_allocated" not in df.columns or "direction" not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="🚦 Signal Phase Timeline (Waiting for data...)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=200,
        )
        return fig

    directions  = df["direction"].tolist()
    green_times = df["green_time_allocated"].tolist()
    bar_colors  = [DIR_COLORS.get(str(d), "#90CAF9") for d in directions]

    fig = go.Figure(go.Bar(
        x=green_times,
        y=[str(d).capitalize() for d in directions],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v}s" for v in green_times],
        textposition="inside",
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(
        title="🚦 Avg Green Time Allocated per Direction",
        xaxis_title="Green Time (seconds)",
        yaxis_title="Direction",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.8)",
        font=dict(color="white"),
        height=300,
        margin=dict(t=50, b=40, l=80, r=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig


def sim_metrics_bar(sim_data: List[Dict]) -> go.Figure:
    """Grouped bar: waiting time and queue length per direction."""
    if not sim_data:
        fig = go.Figure()
        fig.update_layout(
            title="📉 SUMO Metrics (No simulation data)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=280,
        )
        return fig

    df = pd.DataFrame(sim_data)
    directions = df.get("direction", pd.Series()).tolist()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Avg Waiting Time (s)", "Avg Queue Length"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Bar(
        x=directions,
        y=df.get("avg_waiting", [0]*len(directions)),
        name="Avg Wait",
        marker_color=[DIR_COLORS.get(d, "#90CAF9") for d in directions],
        text=[f"{v:.1f}s" for v in df.get("avg_waiting", [])],
        textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=directions,
        y=df.get("avg_queue", [0]*len(directions)),
        name="Avg Queue",
        marker_color=[DIR_COLORS.get(d, "#90CAF9") for d in directions],
        text=[f"{v:.1f}" for v in df.get("avg_queue", [])],
        textposition="outside",
    ), row=1, col=2)

    fig.update_layout(
        title="📉 SUMO Simulation Metrics",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.8)",
        font=dict(color="white"),
        showlegend=False,
        height=300,
        margin=dict(t=70, b=40, l=40, r=20),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    return fig


def congestion_heatmap(density_history: List[Dict]) -> go.Figure:
    """Heatmap of congestion score over time per direction."""
    if not density_history:
        fig = go.Figure()
        fig.update_layout(
            title="Congestion Heatmap (No data)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=250,
        )
        return fig

    df = pd.DataFrame(density_history)
    if "direction" not in df.columns or "congestion_score" not in df.columns:
        return go.Figure()

    pivot = df.pivot_table(
        index="direction",
        columns=df.groupby("direction").cumcount(),
        values="congestion_score",
        aggfunc="mean",
    ).fillna(0)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        y=list(pivot.index),
        colorscale=[[0, "#1a472a"], [0.5, "#f59e0b"], [1, "#dc2626"]],
        showscale=True,
        colorbar=dict(
            title=dict(text="Score", font=dict(color="white")),
            tickfont=dict(color="white"),
        ),
        zmin=0, zmax=100,
    ))
    fig.update_layout(
        title="Congestion Score Heatmap (Direction x Time)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.8)",
        font=dict(color="white"),
        yaxis=dict(tickfont=dict(color="white")),
        xaxis=dict(title="Time steps", gridcolor="rgba(255,255,255,0.1)"),
        height=270,
        margin=dict(t=50, b=40, l=80, r=80),
    )
    return fig


def class_distribution_pie(count_by_class: Dict[str, int]) -> go.Figure:
    """Pie chart of vehicle type distribution."""
    cls_colors = {
        "car":        "#4FC3F7",
        "motorcycle": "#FFA726",
        "bus":        "#EF5350",
        "truck":      "#CE93D8",
    }
    labels = [k for k, v in count_by_class.items() if v > 0]
    values = [v for v in count_by_class.values() if v > 0]
    colors = [cls_colors.get(l, "#90CAF9") for l in labels]

    if not labels:
        fig = go.Figure()
        fig.update_layout(
            title="Vehicle Type Distribution (No data)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=280,
        )
        return fig

    fig = go.Figure(go.Pie(
        labels=[l.capitalize() for l in labels],
        values=values,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0.4)", width=1)),
        textinfo="label+percent",
        textfont=dict(color="white"),
        hole=0.4,
    ))
    fig.update_layout(
        title="Vehicle Type Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
        height=300,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig