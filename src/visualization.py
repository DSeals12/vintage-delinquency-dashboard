"""
visualization.py
----------------
Reusable Plotly and Seaborn charting functions for the
vintage delinquency and net loss rate dashboard.

All functions return a Plotly figure object (or Matplotlib axis)
and can be saved to outputs/charts/.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ── Color Palette ─────────────────────────────────────────────────────────────
# One distinct color per vintage — 8 cohorts, accessible palette

VINTAGE_COLORS = [
    "#2E86AB",  # 2022-Q1 — steel blue
    "#A23B72",  # 2022-Q2 — plum
    "#F18F01",  # 2022-Q3 — amber
    "#C73E1D",  # 2022-Q4 — rust
    "#3B1F2B",  # 2023-Q1 — dark burgundy
    "#44BBA4",  # 2023-Q2 — teal
    "#E94F37",  # 2023-Q3 — coral
    "#393E41",  # 2023-Q4 — charcoal
]

DPD_COLORS = {
    "30 DPD":      "#F18F01",
    "60 DPD":      "#C73E1D",
    "90+ DPD":     "#7B2D8B",
    "Charged Off": "#1A1A2E",
}

OUTPUT_DIR = Path("outputs/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Chart 1: Vintage DPD Curves (faceted) ────────────────────────────────────

def plot_vintage_dpd_curves(
    vintage_summary: pd.DataFrame,
    dpd_metric: str = "dpd_30_rate",
    title_suffix: str = "30 DPD",
    save: bool = True,
) -> go.Figure:
    """
    Line chart of DPD rate vs. months on book, one line per vintage.
    Optionally faceted by DPD bucket.
    """
    vintages = sorted(vintage_summary["vintage"].unique())
    fig = go.Figure()

    for i, vintage in enumerate(vintages):
        v = vintage_summary[vintage_summary["vintage"] == vintage].sort_values("months_on_book")
        fig.add_trace(go.Scatter(
            x=v["months_on_book"],
            y=v[dpd_metric] * 100,
            mode="lines",
            name=vintage,
            line=dict(color=VINTAGE_COLORS[i % len(VINTAGE_COLORS)], width=2),
            hovertemplate=(
                f"<b>{vintage}</b><br>"
                "MOB: %{x}<br>"
                f"{title_suffix} Rate: %{{y:.2f}}%<extra></extra>"
            ),
        ))

    # Portfolio average line
    avg = vintage_summary.groupby("months_on_book")[dpd_metric].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=avg["months_on_book"],
        y=avg[dpd_metric] * 100,
        mode="lines",
        name="Portfolio Avg",
        line=dict(color="#AAAAAA", width=2, dash="dash"),
        hovertemplate="Portfolio Avg<br>MOB: %{x}<br>Rate: %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Vintage {title_suffix} Rate by Months on Book", font=dict(size=16)),
        xaxis_title="Months on Book (MOB)",
        yaxis_title=f"{title_suffix} Rate (%)",
        yaxis_ticksuffix="%",
        legend_title="Vintage",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        height=480,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE", dtick=3)
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")

    if save:
        fname = OUTPUT_DIR / f"vintage_{dpd_metric}_curves.png"
        fig.write_image(str(fname), scale=2)
        print(f"Saved: {fname}")

    return fig


# ── Chart 2: All DPD Buckets — Faceted Subplots ───────────────────────────────

def plot_dpd_dashboard(vintage_summary: pd.DataFrame, save: bool = True) -> go.Figure:
    """
    2x2 subplot grid: 30 DPD, 60 DPD, 90 DPD, Net Loss Rate — all vintages.
    """
    metrics = [
        ("dpd_30_rate",  "30 DPD Rate"),
        ("dpd_60_rate",  "60 DPD Rate"),
        ("dpd_90_rate",  "90+ DPD Rate"),
        ("net_loss_rate","Cumul. Net Loss Rate"),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m[1] for m in metrics],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    vintages = sorted(vintage_summary["vintage"].unique())
    show_legend = True

    for idx, (metric, label) in enumerate(metrics):
        row, col = divmod(idx, 2)
        row += 1; col += 1

        for i, vintage in enumerate(vintages):
            v = vintage_summary[vintage_summary["vintage"] == vintage].sort_values("months_on_book")
            fig.add_trace(
                go.Scatter(
                    x=v["months_on_book"],
                    y=v[metric] * 100,
                    mode="lines",
                    name=vintage,
                    legendgroup=vintage,
                    showlegend=show_legend,
                    line=dict(color=VINTAGE_COLORS[i % len(VINTAGE_COLORS)], width=1.5),
                    hovertemplate=f"<b>{vintage}</b><br>MOB: %{{x}}<br>{label}: %{{y:.2f}}%<extra></extra>",
                ),
                row=row, col=col,
            )
        show_legend = False  # only show legend once

    fig.update_layout(
        title=dict(text="Vintage Delinquency & Loss Dashboard", font=dict(size=18)),
        legend_title="Vintage",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        height=640,
        hovermode="x unified",
    )
    for axis in fig.layout:
        if axis.startswith("xaxis"):
            fig.layout[axis].update(showgrid=True, gridcolor="#EEEEEE", dtick=3)
        if axis.startswith("yaxis"):
            fig.layout[axis].update(showgrid=True, gridcolor="#EEEEEE", ticksuffix="%")

    if save:
        fname = OUTPUT_DIR / "dpd_dashboard.png"
        fig.write_image(str(fname), scale=2)
        print(f"Saved: {fname}")

    return fig


# ── Chart 3: Roll Rate Heatmap ────────────────────────────────────────────────

def plot_roll_rate_heatmap(roll_rates: pd.DataFrame, vintage: str = "All", save: bool = True):
    """
    Seaborn heatmap of the transition matrix (from_state → to_state).
    """
    display_states = [s for s in roll_rates.index if s not in ("paid_off",)]
    matrix = roll_rates.loc[
        [s for s in display_states if s in roll_rates.index],
        [s for s in display_states if s in roll_rates.columns],
    ]

    labels = {
        "current":     "Current",
        "dpd_30":      "30 DPD",
        "dpd_60":      "60 DPD",
        "dpd_90":      "90 DPD",
        "dpd_120":     "120 DPD",
        "charged_off": "Charged Off",
    }
    matrix.index   = [labels.get(s, s) for s in matrix.index]
    matrix.columns = [labels.get(s, s) for s in matrix.columns]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        matrix * 100,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="#DDDDDD",
        cbar_kws={"label": "Transition Rate (%)"},
        ax=ax,
    )
    ax.set_title(f"Roll Rate Transition Matrix — {vintage}", fontsize=14, pad=12)
    ax.set_xlabel("To State", fontsize=11)
    ax.set_ylabel("From State", fontsize=11)
    plt.tight_layout()

    if save:
        fname = OUTPUT_DIR / f"roll_rate_heatmap_{vintage.replace(' ', '_')}.png"
        plt.savefig(str(fname), dpi=150, bbox_inches="tight")
        print(f"Saved: {fname}")
        plt.close()

    return ax


# ── Chart 4: Net Loss Rate — Vintage Comparison ───────────────────────────────

def plot_net_loss_curves(vintage_summary: pd.DataFrame, save: bool = True) -> go.Figure:
    """
    Cumulative net loss rate curves by vintage with portfolio average band.
    """
    vintages = sorted(vintage_summary["vintage"].unique())

    fig = go.Figure()

    # Confidence band: min/max across vintages
    grouped = vintage_summary.groupby("months_on_book")["net_loss_rate"]
    mob_range = vintage_summary["months_on_book"].unique()
    fig.add_trace(go.Scatter(
        x=list(mob_range) + list(mob_range[::-1]),
        y=list(grouped.max() * 100) + list(grouped.min()[::-1] * 100),
        fill="toself",
        fillcolor="rgba(180,180,180,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        name="Vintage range",
        hoverinfo="skip",
    ))

    for i, vintage in enumerate(vintages):
        v = vintage_summary[vintage_summary["vintage"] == vintage].sort_values("months_on_book")
        fig.add_trace(go.Scatter(
            x=v["months_on_book"],
            y=v["net_loss_rate"] * 100,
            mode="lines",
            name=vintage,
            line=dict(color=VINTAGE_COLORS[i % len(VINTAGE_COLORS)], width=2),
            hovertemplate=(
                f"<b>{vintage}</b><br>"
                "MOB: %{x}<br>"
                "Net Loss Rate: %{y:.2f}%<extra></extra>"
            ),
        ))

    avg = vintage_summary.groupby("months_on_book")["net_loss_rate"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=avg["months_on_book"],
        y=avg["net_loss_rate"] * 100,
        mode="lines",
        name="Portfolio Avg",
        line=dict(color="#333333", width=2.5, dash="dot"),
    ))

    fig.update_layout(
        title=dict(text="Cumulative Net Loss Rate by Vintage (% of Origination Balance)", font=dict(size=16)),
        xaxis_title="Months on Book (MOB)",
        yaxis_title="Cumulative Net Loss Rate (%)",
        yaxis_ticksuffix="%",
        legend_title="Vintage",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        height=480,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE", dtick=3)
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")

    if save:
        fname = OUTPUT_DIR / "net_loss_rate_curves.png"
        fig.write_image(str(fname), scale=2)
        print(f"Saved: {fname}")

    return fig


# ── Chart 5: Cohort Snapshot Bar — Vintage Comparison at Fixed MOB ────────────

def plot_cohort_snapshot(comparison_df: pd.DataFrame, mob: int, metric: str = "net_loss_rate", save: bool = True) -> go.Figure:
    """
    Bar chart comparing vintages at a fixed MOB snapshot.
    Highlights vintages above portfolio average.
    """
    avg = comparison_df[metric].mean()
    colors = [
        "#C73E1D" if v > avg else "#2E86AB"
        for v in comparison_df[metric]
    ]

    fig = go.Figure(go.Bar(
        x=comparison_df["vintage"],
        y=comparison_df[metric] * 100,
        marker_color=colors,
        text=(comparison_df[metric] * 100).round(2).astype(str) + "%",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Rate: %{y:.2f}%<extra></extra>",
    ))

    fig.add_hline(
        y=avg * 100,
        line_dash="dash",
        line_color="#666666",
        annotation_text=f"Portfolio avg: {avg*100:.2f}%",
        annotation_position="top right",
    )

    label = metric.replace("_", " ").title()
    fig.update_layout(
        title=dict(text=f"{label} at MOB {mob} — Vintage Comparison", font=dict(size=16)),
        xaxis_title="Origination Vintage",
        yaxis_title=f"{label} (%)",
        yaxis_ticksuffix="%",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        height=420,
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")

    if save:
        fname = OUTPUT_DIR / f"cohort_snapshot_mob{mob}_{metric}.png"
        fig.write_image(str(fname), scale=2)
        print(f"Saved: {fname}")

    return fig
