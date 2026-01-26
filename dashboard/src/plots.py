#src/plots.py
from __future__ import annotations

import altair as alt
import pandas as pd


def _downsample(df: pd.DataFrame, max_points: int = 5000, seed: int = 0) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def bar_state_distribution(state_dist: pd.DataFrame, title: str) -> alt.Chart:
    if not {"State", "Percent"}.issubset(state_dist.columns):
        raise ValueError(f"bar_state_distribution: expected columns State, Percent. Got: {list(state_dist.columns)}")

    plot_df = state_dist.copy()
    plot_df["State"] = plot_df["State"].astype(str)
    plot_df["Percent"] = pd.to_numeric(plot_df["Percent"], errors="coerce").fillna(0.0)

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y("State:N", sort="-x", title=None),
            x=alt.X("Percent:Q", title="Percentage (%)"),
            tooltip=[alt.Tooltip("State:N"), alt.Tooltip("Percent:Q", format=".1f")],
        )
        .properties(height=260, title=title)
    )


def scatter_load_vs_sleep(bs_merged: pd.DataFrame, title: str) -> alt.Chart:
    d = bs_merged.dropna(subset=["mean_prb", "p_sleep_cells"]).copy()
    d = _downsample(d, max_points=5000, seed=0)

    return (
        alt.Chart(d)
        .mark_circle(size=60)
        .encode(
            x=alt.X("mean_prb:Q", title="Mean PRB Usage Ratio (%)"),
            y=alt.Y("p_sleep_cells:Q", title="Fraction of rows sleeping (GT)"),
            tooltip=[
                "Base Station ID",
                alt.Tooltip("mean_prb:Q", format=".2f"),
                alt.Tooltip("p_sleep_cells:Q", format=".3f"),
                "State",
            ],
        )
        .properties(height=320, title=title)
    )


def topn_scatter(top: pd.DataFrame, title: str) -> alt.Chart:
    return (
        alt.Chart(top)
        .mark_circle()
        .encode(
            x=alt.X("baseline_kWh:Q", title="Baseline energy (kWh)"),
            y=alt.Y("eco_saved_pct:Q", title="Eco saved (% of baseline)"),
            size=alt.Size("eco_saved_kWh:Q", title="Eco saved (kWh)"),
            color=alt.Color("p_sleep:Q", title="GT sleep fraction (5G)"),
            tooltip=[
                alt.Tooltip("Base Station ID:N"),
                alt.Tooltip("traffic_kbyte:Q", format=",.0f", title="Traffic (KByte)"),
                alt.Tooltip("mean_prb:Q", format=".2f", title="Mean PRB (%)"),
                alt.Tooltip("baseline_kWh:Q", format=",.3f", title="Baseline (kWh)"),
                alt.Tooltip("eco_saved_kWh:Q", format=",.3f", title="Eco saved (kWh)"),
                alt.Tooltip("eco_saved_pct:Q", format=".2f", title="Eco saved (%)"),
                alt.Tooltip("p_sleep:Q", format=".3f", title="GT sleep fraction"),
            ],
        )
        .properties(height=420, title=title)
    )


def cell_gt_sleep_bar(gt_melt: pd.DataFrame, title: str) -> alt.Chart:
    return (
        alt.Chart(gt_melt)
        .mark_bar()
        .encode(
            y=alt.Y("Cell ID:N", sort="-x", title="Cell ID"),
            x=alt.X("Value:Q", title="Value"),
            color=alt.Color("Metric:N", legend=alt.Legend(title=None)),
            tooltip=["Cell ID", "Metric", alt.Tooltip("Value:Q", format=".3f")],
        )
        .properties(title=title)
    )


def cell_eco_savings_bar(sav: pd.DataFrame, title: str) -> alt.Chart:
    return (
        alt.Chart(sav)
        .mark_bar()
        .encode(
            y=alt.Y("Cell ID:N", sort="-x", title="Cell ID"),
            x=alt.X("eco_saved_pct_of_baseline:Q", title="Eco saved (% baseline energy)"),
            tooltip=[
                "Cell ID",
                alt.Tooltip("eco_saved_pct_of_baseline:Q", format=".2f"),
                alt.Tooltip("eco_saved_kWh:Q", format=",.3f", title="Eco saved (kWh)"),
            ],
        )
        .properties(title=title)
    )


def hist_numeric(df: pd.DataFrame, col: str, title: str, bin_step: float | None = None) -> alt.Chart:
    plot_df = df[[col]].copy()
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna()

    b = alt.Bin(step=bin_step) if bin_step is not None else alt.Bin(maxbins=30)

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:Q", bin=b, title=col),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Count")],
        )
        .properties(height=260, title=title)
    )


def scatter_xy(df: pd.DataFrame, x: str, y: str, title: str, tooltip_cols: list[str]) -> alt.Chart:
    plot_df = df.copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])

    plot_df = _downsample(plot_df, max_points=5000, seed=0)

    tooltips = []
    for c in tooltip_cols:
        if c in plot_df.columns:
            tooltips.append(alt.Tooltip(f"{c}:N"))
    tooltips += [alt.Tooltip(f"{x}:Q", format=".3f"), alt.Tooltip(f"{y}:Q", format=".3f")]

    return (
        alt.Chart(plot_df)
        .mark_circle(size=55)
        .encode(
            x=alt.X(f"{x}:Q", title=x),
            y=alt.Y(f"{y}:Q", title=y),
            tooltip=tooltips,
        )
        .properties(height=320, title=title)
    )
