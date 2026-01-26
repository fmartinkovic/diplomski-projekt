# src/ui_tabs.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import resolve_col
from src.aggregations import bs_summary, bs_mean_prb, state_distribution
from src.kpis import bs_level_gt_vs_pred
from src.policy import compute_thresholds, decide_state
from src import plots

from src.cache import get_bs_summary_cached, get_per_cell_kpis_cached


def _to_kwh(x_wh: float) -> float:
    return float(x_wh) / 1000.0


def _eco_vs_gt_metrics_bs(df_view: pd.DataFrame) -> dict:
    """
    For ML controller:
      - predicted positive: any ECO rows for a BS
      - GT positive: any sleep_on rows for a BS (5G ground truth)
    """
    bs_col = resolve_col(df_view, "bs")
    if "sleep_on" not in df_view.columns or "State" not in df_view.columns:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    by_bs = df_view.groupby(bs_col, dropna=False).agg(
        gt_sleep_any=("sleep_on", lambda s: bool((s == True).any())),
        pred_eco_any=("State", lambda s: bool((s == "ECO").any())),
    ).reset_index()

    tp = int(((by_bs["pred_eco_any"]) & (by_bs["gt_sleep_any"])).sum())
    fp = int(((by_bs["pred_eco_any"]) & (~by_bs["gt_sleep_any"])).sum())
    fn = int(((~by_bs["pred_eco_any"]) & (by_bs["gt_sleep_any"])).sum())
    tn = int(((~by_bs["pred_eco_any"]) & (~by_bs["gt_sleep_any"])).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def render_tab_overview(
    df_view: pd.DataFrame,
    df_policy_full: pd.DataFrame,
    tod: str,
    dataset: str,
    window_label: str,
    controller_type: str,
    threshold_scope: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    ml_tau_on: float,
    ml_tau_off: float,
    ml_hysteresis_enabled: bool,
    show_gt_metrics: bool = True,
) -> None:
    st.subheader(f"Global Network Overview ({window_label})")

    bs_col = resolve_col(df_view, "bs")
    cell_col = resolve_col(df_view, "cell")
    traffic_col = resolve_col(df_view, "traffic_kb")

    total_bs = int(df_view[bs_col].nunique())
    total_cells = int(df_view[cell_col].nunique())
    total_traffic_kbyte = float(pd.to_numeric(df_view[traffic_col], errors="coerce").sum(skipna=True))

    total_baseline_kwh = _to_kwh(float(pd.to_numeric(df_view["baseline_Wh"], errors="coerce").sum(skipna=True)))
    total_saved_kwh = _to_kwh(float(pd.to_numeric(df_view["eco_saved_Wh"], errors="coerce").sum(skipna=True)))
    total_saved_pct = (100.0 * total_saved_kwh / total_baseline_kwh) if total_baseline_kwh > 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Base Stations", total_bs)
    c2.metric("Cells", total_cells)
    c3.metric("Total Traffic (KByte)", f"{total_traffic_kbyte:,.0f}")
    c4.metric("Baseline energy (kWh)", f"{total_baseline_kwh:,.3f}")
    c5.metric("Eco saved (kWh)", f"{total_saved_kwh:,.3f} ({total_saved_pct:.2f}%)")

    json_summary = {
        "dataset": dataset,
        "time_of_day": tod,
        "window": window_label,
        "base_stations": total_bs,
        "cells": total_cells,
        "total_traffic_kbyte": total_traffic_kbyte,
        "energy_kwh": {
            "baseline_kwh": total_baseline_kwh,
            "eco_saved_kwh": total_saved_kwh,
            "eco_saved_pct": total_saved_pct,
        },
        "controller": {
            "type": controller_type,
            "alpha": float(alpha),
        }
    }

    if controller_type == "Heuristic":
        # --- BS mean PRB
        prb_col = resolve_col(df_view, "prb")
        bs_mean = bs_mean_prb(df_view)

        # --- Thresholds for interpretability on BS mean
        if threshold_scope == "Global":
            p30 = pd.to_numeric(df_view["p30"], errors="coerce").dropna() if "p30" in df_view.columns else pd.Series(dtype=float)
            p70 = pd.to_numeric(df_view["p70"], errors="coerce").dropna() if "p70" in df_view.columns else pd.Series(dtype=float)
            bs_mean["p30"] = float(p30.iloc[0]) if len(p30) else float("nan")
            bs_mean["p70"] = float(p70.iloc[0]) if len(p70) else float("nan")
        elif threshold_scope == "Per-BaseStation":
            thr_bs = df_view.groupby(bs_col, dropna=False)[["p30", "p70"]].first().reset_index()
            bs_mean = bs_mean.merge(thr_bs, on=bs_col, how="left")
        else:
            p30g, p70g = compute_thresholds(df_view[prb_col])
            bs_mean["p30"] = p30g
            bs_mean["p70"] = p70g

        bs_mean["State"] = bs_mean.apply(
            lambda r: decide_state(r["mean_prb"], r.get("p30", np.nan), r.get("p70", np.nan)),
            axis=1,
        ).astype(str)

        dist = state_distribution(bs_mean["State"])
        st.altair_chart(
            plots.bar_state_distribution(dist, title="Base Station State Distribution (heuristic)"),
            use_container_width=True,
        )

        if show_gt_metrics:
            gt_metrics = bs_level_gt_vs_pred(df_view, state_col="State") if "State" in df_view.columns else {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            st.markdown("##### Ground Truth (5G Only)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("TP (SLEEP & GT)", gt_metrics["tp"])
            m2.metric("FP (SLEEP & !GT)", gt_metrics["fp"])
            m3.metric("FN (!SLEEP & GT)", gt_metrics["fn"])
            m4.metric("TN (!SLEEP & !GT)", gt_metrics["tn"])

            sleep_by_bs = df_view.groupby(bs_col, dropna=False)["sleep_on"].mean().reset_index().rename(columns={"sleep_on": "p_sleep_cells"})
            merged = bs_mean.merge(sleep_by_bs, on=bs_col, how="left")
            st.altair_chart(
                plots.scatter_load_vs_sleep(merged, title="Load vs GT deep sleep (per Base Station)"),
                use_container_width=True,
            )
            json_summary["heuristic_vs_gt_bs_level"] = gt_metrics

        st.markdown("**Summary (JSON)**")
        st.json(json_summary)
        return

    # -------------------------
    # ML overview
    # -------------------------
    dist_rows = state_distribution(df_view["State"].astype(str))
    st.altair_chart(
        plots.bar_state_distribution(dist_rows, title="Row-level State Distribution (ML controller)"),
        use_container_width=True,
    )

    if show_gt_metrics:
        gt_proxy = _eco_vs_gt_metrics_bs(df_view)
        st.markdown("##### Ground Truth (5G Only)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TP (ECO & GT)", gt_proxy["tp"])
        m2.metric("FP (ECO & !GT)", gt_proxy["fp"])
        m3.metric("FN (!ECO & GT)", gt_proxy["fn"])
        m4.metric("TN (!ECO & !GT)", gt_proxy["tn"])
        json_summary["eco_vs_gt_sleep_bs_level"] = gt_proxy

    # Optional: show probability summary if present
    if "p_sleep_on" in df_view.columns:
        p = pd.to_numeric(df_view["p_sleep_on"], errors="coerce").dropna()
        if len(p):
            s1, s2, s3 = st.columns(3)
            s1.metric("p_sleep_on mean", f"{float(p.mean()):.4f}")
            s2.metric("p_sleep_on p90", f"{float(p.quantile(0.9)):.4f}")
            s3.metric("p_sleep_on p99", f"{float(p.quantile(0.99)):.4f}")

    st.markdown("**Summary (JSON)**")
    st.json(json_summary)


def render_tab_topn(
    df_view: pd.DataFrame,
    tod: str,
    dataset: str,
    window_label: str,
    *,
    path_str: str,
    controller_type: str,
    threshold_scope: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    ml_model_path: str,
    ml_tau_on: float,
    ml_tau_off: float,
    ml_hysteresis_enabled: bool,
    ml_use_energy_features: bool,
    ml_use_prev_features: bool,
    ml_use_time_features: bool,
    ml_use_time_cyclical: bool,
    window_mode: str,
    tod_bin: int,
    show_gt_metrics: bool = True,
) -> None:
    st.subheader(f"Top-N Base Stations ({window_label})")

    summary = get_bs_summary_cached(
        path_str,
        controller_type=controller_type,
        threshold_scope=threshold_scope,
        alpha=alpha,
        hysteresis_enabled=hysteresis_enabled,
        h_sleep=h_sleep,
        h_eco=h_eco,
        ml_model_path=ml_model_path,
        ml_tau_on=ml_tau_on,
        ml_tau_off=ml_tau_off,
        ml_hysteresis_enabled=ml_hysteresis_enabled,
        ml_use_energy_features=ml_use_energy_features,
        ml_use_prev_features=ml_use_prev_features,
        ml_use_time_features=ml_use_time_features,
        ml_use_time_cyclical=ml_use_time_cyclical,
        window_mode=window_mode,
        tod_bin=tod_bin,
    )

    bs_col = resolve_col(summary, "bs")

    summary = summary.copy()
    summary["baseline_kWh"] = summary["baseline_Wh"].map(_to_kwh)
    summary["eco_saved_kWh"] = summary["eco_saved_Wh"].map(_to_kwh)

    # Filtering options
    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    with c1:
        top_n = st.slider("Select N", 5, 50, 10)
    with c2:
        options = ["eco_saved_kWh", "eco_saved_pct", "baseline_kWh", "traffic_kbyte"]
        if show_gt_metrics:
            options.append("p_sleep")
        rank_by = st.selectbox("Rank by", options, index=0)
    with c3:
        ascending = st.checkbox("Ascending", value=False)

    top = summary.sort_values(rank_by, ascending=ascending).head(top_n).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))

    st.markdown("### A) Ranked KPI table (Top-N)")
    display_cols = [
        "rank",
        bs_col,
        "traffic_kbyte",
        "mean_prb",
        "baseline_kWh",
        "eco_saved_kWh",
        "eco_saved_pct",
    ]
    if show_gt_metrics:
        display_cols.append("p_sleep")

    top_disp = top[display_cols].rename(
        columns={
            bs_col: "Base Station ID",
            "traffic_kbyte": "traffic (KByte)",
            "mean_prb": "mean PRB (%)",
            "baseline_kWh": "baseline energy (kWh)",
            "eco_saved_kWh": "eco saved (kWh)",
            "eco_saved_pct": "eco saved (%)",
            "p_sleep": "GT sleep fraction",
        }
    )

    fmt = {
        "traffic (KByte)": "{:,.0f}",
        "mean PRB (%)": "{:.2f}",
        "baseline energy (kWh)": "{:,.3f}",
        "eco saved (kWh)": "{:,.3f}",
        "eco saved (%)": "{:.2f}",
    }
    if show_gt_metrics:
        fmt["GT sleep fraction"] = "{:.3f}"

    styler = (
        top_disp.style.format(fmt)
        .bar(subset=["eco saved (kWh)"], align="zero")
        .bar(subset=["eco saved (%)"], vmin=0, vmax=max(1e-9, float(top_disp["eco saved (%)"].max())))
    )
    st.dataframe(styler, use_container_width=True)

    st.markdown("### B) Impact–feasibility scatter (Top-N)")
    plot_top = top.rename(columns={bs_col: "Base Station ID"}).copy()
    
    # If no GT, fill p_sleep with 0 for plot compatibility (it won't be informative but won't crash)
    if not show_gt_metrics and "p_sleep" not in plot_top.columns:
        plot_top["p_sleep"] = 0.0

    st.altair_chart(
        plots.topn_scatter(plot_top, title="Top-N: savings impact vs savings intensity"),
        use_container_width=True,
    )


def render_tab_drilldown(
    df_view: pd.DataFrame,
    tod: str,
    dataset: str,
    window_label: str,
    controller_type: str,
    threshold_scope: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    show_gt_metrics: bool = True,
) -> None:
    from src.kpis import per_cell_kpis

    st.subheader(f"Base Station Drill-Down ({window_label})")

    bs_col = resolve_col(df_view, "bs")
    traffic_col = resolve_col(df_view, "traffic_kb")

    top_bs_for_selector = (
        df_view.groupby(bs_col, dropna=False)[traffic_col]
        .sum()
        .nlargest(25)
        .reset_index()
    )
    if top_bs_for_selector.empty:
        st.info("No base stations available for this window.")
        return

    selected_bs = st.selectbox("Select Base Station", top_bs_for_selector[bs_col])
    df_bs = df_view.loc[df_view[bs_col] == selected_bs].copy()
    if df_bs.empty:
        st.info("No data for the selected base station in this window.")
        return

    cell_kpi = per_cell_kpis(df_bs).sort_values("eco_saved_Wh", ascending=False).reset_index(drop=True)
    cell_kpi = cell_kpi.copy()
    cell_kpi["baseline_kWh"] = cell_kpi["baseline_Wh"].map(_to_kwh)
    cell_kpi["eco_saved_kWh"] = cell_kpi["eco_saved_Wh"].map(_to_kwh)

    st.markdown("### Per-cell KPIs")
    
    # Hide GT columns if irrelevant
    disp_cols = list(cell_kpi.columns)
    if not show_gt_metrics:
        # These are generated by per_cell_kpis, but are 0 in 4G
        cols_to_hide = ["p_sleep", "f_sleep", "n_bouts", "bouts_per_day", "mean_bout_intervals", "mean_bout_minutes"]
        disp_cols = [c for c in disp_cols if c not in cols_to_hide]
    
    st.dataframe(cell_kpi[disp_cols], use_container_width=True)

    left, right = st.columns(2)

    with left:
        if show_gt_metrics:
            gt_sleep = cell_kpi[["Cell ID", "p_sleep", "f_sleep"]].copy()
            gt_sleep = gt_sleep.rename(columns={"p_sleep": "P(sleep_on)", "f_sleep": "Sleep time fraction"})
            gt_melt = gt_sleep.melt(id_vars=["Cell ID"], var_name="Metric", value_name="Value")
            chart = plots.cell_gt_sleep_bar(gt_melt, title="Ground-truth deep sleep metrics by cell")
            st.altair_chart(chart.properties(height=min(520, 22 * len(gt_sleep) + 120)), use_container_width=True)
        else:
            st.info("Ground-truth sleep metrics unavailable (4G Inference).")

    with right:
        sav = cell_kpi[["Cell ID", "eco_saved_pct_of_baseline", "eco_saved_kWh"]].copy()
        sav = sav.sort_values("eco_saved_kWh", ascending=False)
        chart = plots.cell_eco_savings_bar(sav, title="Economy Mode savings by cell")
        st.altair_chart(chart.properties(height=min(520, 22 * len(sav) + 120)), use_container_width=True)

    baseline_total_kwh = float(pd.to_numeric(df_bs["baseline_Wh"], errors="coerce").sum(skipna=True)) / 1000.0
    eco_saved_total_kwh = float(pd.to_numeric(df_bs["eco_saved_Wh"], errors="coerce").sum(skipna=True)) / 1000.0
    eco_saved_pct = (100.0 * eco_saved_total_kwh / baseline_total_kwh) if baseline_total_kwh > 0 else 0.0

    st.markdown("### Selected Base Station Summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Baseline energy (kWh)", f"{baseline_total_kwh:,.3f}")
    s2.metric("Eco saved (kWh)", f"{eco_saved_total_kwh:,.3f}")
    s3.metric("Eco saved (% baseline)", f"{eco_saved_pct:.2f}%")


def render_tab_heterogeneity(
    df_view: pd.DataFrame,
    tod: str,
    dataset: str,
    window_label: str,
    controller_type: str,
    threshold_scope: str,
    *,
    path_str: str,
    alpha: float,
    hysteresis_enabled: bool,
    h_sleep: float,
    h_eco: float,
    ml_model_path: str,
    ml_tau_on: float,
    ml_tau_off: float,
    ml_hysteresis_enabled: bool,
    ml_use_energy_features: bool,
    ml_use_prev_features: bool,
    ml_use_time_features: bool,
    ml_use_time_cyclical: bool,
    window_mode: str,
    tod_bin: int,
    show_gt_metrics: bool = True,
) -> None:
    st.subheader(f"Heterogeneity ({window_label})")

    st.caption("Tip: This tab can be expensive. Use the button to compute cached heterogeneity KPIs.")
    compute = st.button("Compute heterogeneity KPIs (cached)", type="primary")
    if not compute:
        st.info("Click the button to compute per-cell and per-base-station heterogeneity KPIs.")
        return

    cell_kpi = get_per_cell_kpis_cached(
        path_str,
        controller_type=controller_type,
        threshold_scope=threshold_scope,
        alpha=alpha,
        hysteresis_enabled=hysteresis_enabled,
        h_sleep=h_sleep,
        h_eco=h_eco,
        ml_model_path=ml_model_path,
        ml_tau_on=ml_tau_on,
        ml_tau_off=ml_tau_off,
        ml_hysteresis_enabled=ml_hysteresis_enabled,
        ml_use_energy_features=ml_use_energy_features,
        ml_use_prev_features=ml_use_prev_features,
        ml_use_time_features=ml_use_time_features,
        ml_use_time_cyclical=ml_use_time_cyclical,
        window_mode=window_mode,
        tod_bin=tod_bin,
    )
    if cell_kpi.empty:
        st.info("No per-cell KPIs available for this window.")
        return

    cell_kpi = cell_kpi.copy()
    cell_kpi["baseline_kWh"] = cell_kpi["baseline_Wh"].map(_to_kwh)
    cell_kpi["eco_saved_kWh"] = cell_kpi["eco_saved_Wh"].map(_to_kwh)

    bs_kpi = get_bs_summary_cached(
        path_str,
        controller_type=controller_type,
        threshold_scope=threshold_scope,
        alpha=alpha,
        hysteresis_enabled=hysteresis_enabled,
        h_sleep=h_sleep,
        h_eco=h_eco,
        ml_model_path=ml_model_path,
        ml_tau_on=ml_tau_on,
        ml_tau_off=ml_tau_off,
        ml_hysteresis_enabled=ml_hysteresis_enabled,
        ml_use_energy_features=ml_use_energy_features,
        ml_use_prev_features=ml_use_prev_features,
        ml_use_time_features=ml_use_time_features,
        ml_use_time_cyclical=ml_use_time_cyclical,
        window_mode=window_mode,
        tod_bin=tod_bin,
    )
    bs_col = resolve_col(bs_kpi, "bs")
    bs_kpi = bs_kpi.copy()
    bs_kpi["baseline_kWh"] = bs_kpi["baseline_Wh"].map(_to_kwh)
    bs_kpi["eco_saved_kWh"] = bs_kpi["eco_saved_Wh"].map(_to_kwh)

    def _cv(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        m = float(s.mean()) if len(s) else 0.0
        sd = float(s.std(ddof=0)) if len(s) else 0.0
        return float(sd / m) if m > 0 else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cells", int(len(cell_kpi)))
    c2.metric("Base Stations", int(len(bs_kpi)))
    if show_gt_metrics:
        c3.metric("CV(p_sleep) across cells", f"{_cv(cell_kpi['p_sleep']):.3f}")
    c4.metric("CV(eco_saved_kWh) across BS", f"{_cv(bs_kpi['eco_saved_kWh']):.3f}")

    st.markdown("### A) Per-cell heterogeneity")
    
    a1, a2 = st.columns(2)
    with a1:
        if show_gt_metrics:
            st.altair_chart(
                plots.hist_numeric(cell_kpi, "p_sleep", "Cells: distribution of P(sleep_on)", bin_step=0.05),
                use_container_width=True,
            )
        else:
             st.altair_chart(
                plots.hist_numeric(cell_kpi, "mean_prb", "Cells: distribution of mean PRB (%)", bin_step=2.0),
                use_container_width=True,
            )
    with a2:
        if show_gt_metrics:
            st.altair_chart(
                plots.hist_numeric(cell_kpi, "mean_prb", "Cells: distribution of mean PRB (%)", bin_step=2.0),
                use_container_width=True,
            )
        else:
             # Just show placeholder or empty
             st.info("p_sleep distribution unavailable (4G Inference).")

    if show_gt_metrics:
        b1, b2 = st.columns(2)
        with b1:
            st.altair_chart(
                plots.hist_numeric(cell_kpi, "mean_bout_minutes", "Cells: mean deep-sleep bout duration (minutes)", bin_step=15.0),
                use_container_width=True,
            )
        with b2:
            st.altair_chart(
                plots.hist_numeric(cell_kpi, "bouts_per_day", "Cells: deep-sleep bouts per day", bin_step=0.5),
                use_container_width=True,
            )

        st.altair_chart(
            plots.scatter_xy(
                cell_kpi,
                x="mean_prb",
                y="p_sleep",
                title="Cells: mean PRB vs P(sleep_on) (downsampled if large)",
                tooltip_cols=["Base Station ID", "Cell ID"],
            ),
            use_container_width=True,
        )

    st.markdown("### B) Per-base-station heterogeneity")
    d1, d2 = st.columns(2)
    with d1:
        st.altair_chart(
            plots.hist_numeric(bs_kpi, "eco_saved_kWh", "Base Stations: distribution of eco saved (kWh)", bin_step=None),
            use_container_width=True,
        )
    with d2:
        st.altair_chart(
            plots.hist_numeric(bs_kpi, "eco_saved_pct", "Base Stations: distribution of eco saved (%)", bin_step=1.0),
            use_container_width=True,
        )

    st.altair_chart(
        plots.scatter_xy(
            bs_kpi.rename(columns={bs_col: "Base Station ID"}),
            x="mean_prb",
            y="eco_saved_pct",
            title="Base Stations: mean PRB vs eco saved (%) (downsampled if large)",
            tooltip_cols=["Base Station ID"],
        ),
        use_container_width=True,
    )

def render_tab_risk_optimization(
    df_view: pd.DataFrame,
    df_policy_full: pd.DataFrame, # Need full dataset for accurate sweep
    alpha: float,
    current_tau_on: float,
    current_tau_off: float,
    prb_threshold: float = 20.0
) -> None:
    from src.kpis import calculate_risk_metrics
    from src.optimization import run_pareto_sweep
    import altair as alt

    st.subheader("⚖️ Risk vs. Savings Optimization")
    
    st.markdown(
        """
        **The Trade-off:** Aggressive energy saving (low thresholds) increases the risk of 
        putting active cells into ECO mode. This tool helps find the 'Sweet Spot'.
        """
    )

    prb_threshold = st.slider(
        "PRB Risk Threshold (%)", 
        min_value=1.0, 
        max_value=50.0, 
        value=20.0, 
        step=1.0,
        help="The load level above which Economy Mode is considered a performance risk."
    )

    # 1. Current Risk Status
    st.markdown("#### 1. Current Risk Profile")
    risk_metrics = calculate_risk_metrics(df_view, prb_threshold=prb_threshold)
    
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Risk Threshold (PRB)", 
        f"> {prb_threshold}%", 
        help="Cells with load above this value should NOT be in ECO."
    )
    c2.metric(
        "High Risk Intervals", 
        f"{risk_metrics['risk_intervals']:,}", 
        help="Number of 30-min intervals where State=ECO and Load > Threshold"
    )
    c3.metric(
        "Risk % (of Total Time)", 
        f"{risk_metrics['risk_percent_total']:.2f}%",
        delta_color="inverse",
        delta=None # You could add a reference target here
    )

    if risk_metrics['risk_percent_total'] > 1.0:
        st.warning(f"⚠️ High Risk detected! {risk_metrics['risk_percent_total']:.2f}% of operations coincide with high load.")
    else:
        st.success("✅ Operational Risk is within safe limits (< 1%).")

    st.divider()

    # 2. Pareto Sweep
    st.markdown("#### 2. Optimization Sweep (Pareto Frontier)")
    
    col_run, col_conf = st.columns([1, 3])
    with col_conf:
        sweep_range = st.slider("Sweep range for τ_on", 0.5, 0.95, (0.6, 0.9))
        fixed_hysteresis = st.slider("Fixed Hysteresis (τ_on - τ_off)", 0.0, 0.2, 0.1, 0.05)
    
    with col_run:
        st.write("") # Spacer
        run_opt = st.button("Run Optimization Sweep", type="primary")

    if run_opt:
        # Prepare grid
        t_min, t_max = sweep_range
        grid = np.linspace(t_min, t_max, 15) # 15 steps
        
        # Run sweep on the FULL dataset (df_policy_full) for statistical significance
        # Make sure it has p_sleep_on
        if "p_sleep_on" not in df_policy_full.columns:
            st.error("Cannot run optimization: 'p_sleep_on' missing from data. Ensure ML Controller is active.")
            return

        res_df = run_pareto_sweep(
            df_policy_full, 
            alpha=alpha, 
            tau_on_range=grid, 
            tau_off_offset=fixed_hysteresis, 
            prb_threshold=prb_threshold
        )
        
        # Plot
        base = alt.Chart(res_df).encode(
            tooltip=[
                alt.Tooltip("tau_on", format=".2f"),
                alt.Tooltip("saved_kwh", format=".1f"),
                alt.Tooltip("risk_pct", format=".2f"),
                alt.Tooltip("eco_coverage_pct", format=".1f", title="Eco Coverage %")
            ]
        )

        scatter = base.mark_circle(size=100).encode(
            x=alt.X("risk_pct:Q", title="Risk % (Load > Threshold in ECO)"),
            y=alt.Y("saved_kwh:Q", title="Total Energy Saved (kWh)"),
            color=alt.Color("tau_on:Q", scale=alt.Scale(scheme="viridis"), title="Tau On"),
        )

        line = base.mark_line(color="gray", strokeDash=[5, 5]).encode(
            x="risk_pct:Q",
            y="saved_kwh:Q"
        )
        
        # Highlight current setting if within range
        curr_point = pd.DataFrame({
            "risk_pct": [risk_metrics["risk_percent_total"]],
            # Note: Saved kWh in metrics is based on df_view (window), sweep is df_policy (full). 
            # To be comparable, strictly we should sweep df_view or scale. 
            # Let's sweep df_view for consistency in this visual.
            "saved_kwh": [float(df_view["eco_saved_Wh"].sum() / 1000.0)],
            "label": ["Current"]
        })
        
        # If we swept FULL but plotted VIEW metrics, they mismatch.
        # FIX: Let's run the sweep on `df_view` so the chart scales match the current view.
        res_df_view = run_pareto_sweep(
            df_view, 
            alpha=alpha, 
            tau_on_range=grid, 
            tau_off_offset=fixed_hysteresis, 
            prb_threshold=prb_threshold
        )
        
        # Re-plot using View Data
        base_view = alt.Chart(res_df_view).encode(
             tooltip=[
                alt.Tooltip("tau_on", format=".2f"),
                alt.Tooltip("saved_kwh", format=".1f"),
                alt.Tooltip("risk_pct", format=".2f")
            ]
        )
        scatter_view = base_view.mark_circle(size=100).encode(
            x=alt.X("risk_pct:Q", title="Risk % (Load > Threshold in ECO)"),
            y=alt.Y("saved_kwh:Q", title="Energy Saved (kWh) - Selected Window"),
            color=alt.Color("tau_on:Q", scale=alt.Scale(scheme="viridis"), title="Tau On"),
        )
        
        curr_mark = alt.Chart(curr_point).mark_point(shape="diamond", size=200, color="red").encode(
            x="risk_pct:Q",
            y="saved_kwh:Q",
            tooltip=[alt.Tooltip("label")]
        )

        st.altair_chart((scatter_view + curr_mark).interactive(), use_container_width=True)
        
        # Recommendation table
        best_safe = res_df_view[res_df_view["risk_pct"] <= 1.0].sort_values("saved_kwh", ascending=False).head(1)
        if not best_safe.empty:
            rec = best_safe.iloc[0]
            st.info(f"💡 Recommendation: τ_on = {rec['tau_on']:.2f} yields {rec['saved_kwh']:.1f} kWh with only {rec['risk_pct']:.2f}% risk.")
        else:
            st.warning("No configuration found with Risk < 1%. Consider raising thresholds.")

# src/ui_tabs.py

def render_tab_distribution_check(df_view: pd.DataFrame, path_5g_train: str):
    from src.cache import get_prepared_df
    from src.config import resolve_col
    import altair as alt
    
    st.subheader("📊 Feature Distribution Shift Detection")
    st.info("Comparing the current dataset features against the 5G Training baseline (weekday).")

    # Load 5G Training baseline for comparison
    df_ref = get_prepared_df(path_5g_train)
    
    # Resolve canonical column names to ensure compatibility
    cols_map = {
        "PRB Load": resolve_col(df_view, "prb"),
        "Number of Users": resolve_col(df_view, "users"),
        "Traffic (KB)": resolve_col(df_view, "traffic_kb")
    }
    
    selected_label = st.selectbox("Select feature to compare", list(cols_map.keys()))
    col_name = cols_map[selected_label]

    # Prepare data for plotting
    df_ref_plot = df_ref[[col_name]].copy()
    df_ref_plot["Dataset"] = "5G Training (Ref)"
    
    df_curr_plot = df_view[[col_name]].copy()
    df_curr_plot["Dataset"] = "Current Dataset (View)"
    
    plot_df = pd.concat([df_ref_plot, df_curr_plot], axis=0)

    # Render overlaid histograms
    # We use step=None to let Altair decide bins, or specify bin=alt.Bin(maxbins=40)
    hist = alt.Chart(plot_df).mark_bar(opacity=0.5).encode(
        alt.X(f"{col_name}:Q", bin=alt.Bin(maxbins=40), title=selected_label),
        alt.Y("count()", stack=None, title="Frequency"),
        alt.Color("Dataset:N", scale=alt.Scale(domain=["5G Training (Ref)", "Current Dataset (View)"], range=["#4682b4", "#ef553b"]))
    ).properties(height=400).interactive()

    st.altair_chart(hist, use_container_width=True)

    # Statistical Summary
    st.markdown("#### Statistical Comparison")
    stats = pd.DataFrame({
        "Metric": ["Mean", "Std Dev", "Min", "Max"],
        "5G Training": [
            df_ref[col_name].mean(), 
            df_ref[col_name].std(), 
            df_ref[col_name].min(), 
            df_ref[col_name].max()
        ],
        "Current (4G/5G Evaluation)": [
            df_view[col_name].mean(), 
            df_view[col_name].std(), 
            df_view[col_name].min(), 
            df_view[col_name].max()
        ]
    }).set_index("Metric")
    
    st.table(stats.style.format("{:.2f}"))